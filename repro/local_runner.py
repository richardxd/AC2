"""Compose or run the paper entry points on a private, single-node Ray cluster."""
import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"grpo": "07_15_handoff", "ac2": "08_13_tiedq_seed192",
               "prefix": "08_11_ablation1_replay_noq"}
REVISIONS = {"Qwen/Qwen3-4B-Thinking-2507": "768f209d9ea81521153ed38c47d515654e938aea",
             "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}


def judge_accounting():
    db = sqlite3.connect(f"file:{ROOT / 'runs/judge/spend.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT id,state,charged,receipt FROM calls ORDER BY id").fetchall()
    db.close()
    return [{"id": row[0], "state": row[1], "charged_upper_usd": row[2], "receipt": row[3]} for row in rows]


def configure(args):
    run = (ROOT / args.run_dir).resolve()
    assert run.is_relative_to(ROOT / "runs")
    assert args.gpus in range(1, 8) and args.tp > 0 and args.gpus % args.tp == 0
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == args.gpus
    assert "0" not in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if not args.compose_only:
        import torch
        expected = subprocess.check_output(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
        expected = [x.removeprefix("GPU-") for x in expected]
        observed = [str(torch.cuda.get_device_properties(i).uuid) for i in range(args.gpus)]
        assert observed == expected, {"observed": observed, "expected": expected}
        print("GPU_UUID_MAP", json.dumps(dict(zip(os.environ["CUDA_VISIBLE_DEVICES"].split(","), observed))), flush=True)
    batch = args.batch
    assert batch % 2 == 0
    model = Path(os.environ["HF_HUB_CACHE"]) / ("models--" + args.model.replace("/", "--")) / "snapshots" / REVISIONS[args.model]
    assert (model / "config.json").is_file()
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert all((model / name).is_file() for name in set(index["weight_map"].values()))
    env = {
        "NNODES": 1, "N_GPUS_PER_NODE": args.gpus, "SP_RAY_LOCAL": 1,
        "SP_EXPERIMENT_NAME": run.name, "SP_RUN_DATA_DIR": run,
        "SELF_PLAY_DATA_DIR": ROOT / args.data,
        "SP_VAL_MAP": ROOT / "runs/e3/canonical/val_map.json",
        "SELF_PLAY_JUDGE_URL": "http://127.0.0.1:18791/v1",
        "SP_JUDGE_API_MODEL": "deepseek-flash", "SP_JUDGE_MODEL": "deepseek-flash",
        "SP_JUDGE_STANDALONE": 0, "SP_JUDGE_MAX_TOKENS": args.judge_max_tokens,
        "SP_JUDGE_HTTP_TOTAL_TIMEOUT": 540, "SP_JUDGE_MAX_INFLIGHT": 4,
        "SP_REWARD_NUM_WORKERS": 2,
        "SP_ACTOR_MODEL": model, "SP_ROLLOUT_TP": args.tp,
        "SP_TRAIN_BATCH_SIZE": batch if args.method == "grpo" else batch * 2,
        "SP_PPO_MINI_BATCH": batch // 2, "SP_ROLLOUT_N": args.group,
        "SP_LR": "2e-6", "SP_ENTROPY_COEFF": 0,
        "SP_MAX_PROMPT_LEN": 2048, "SP_MAX_RESPONSE_LEN": args.response,
        # With dp7, a 16-sequence minibatch has 2/3 rows per rank. Allow two
        # maximum-length sequences so the synchronized microbatch count is <=2.
        "SP_PPO_MAX_TOKEN_LEN": (2 if args.gpus > 1 else 1) * (args.response + 2048),
        "SP_LOG_PROB_MAX_TOKEN_LEN": args.response + 2048,
        "SP_ACTOR_GPU_MEM_UTIL": .45, "SP_ACTOR_MAX_NUM_SEQS": 16,
        "SP_MAX_NUM_BATCHED_TOKENS": 4096,
        "SP_USE_FUSED_KERNELS": "True", "SP_DP_PAD": 1,
        "SP_TOTAL_STEPS": args.steps, "SP_SAVE_FREQ": args.save_freq,
        "SP_TEST_FREQ": -1 if args.smoke else args.val_freq,
        "SP_VAL_N": args.val_n, "SP_VAL_BEFORE_TRAIN": str(args.val_only or args.initial_val),
        "SP_VAL_ONLY": str(args.val_only), "SP_MAX_CKPT_KEEP": 100000,
        "SP_KEEP_BEST_CKPT": 0, "SP_CKPT_KEEP_EVERY": 1,
        "SP_ROLLOUT_BACKFILL": 0, "SP_LENPEN_ENABLE": 0, "SP_DIFF_SAMPLING": 0,
        "SP_ADAPTIVE_ENTROPY": 1, "SP_AEC_TARGET_H": .28, "SP_AEC_DELTA": .02,
        "SP_AEC_KMAX": .08, "SP_AEC_KMIN": -.08, "SP_AEC_KINIT": .06,
        "HF_HUB_OFFLINE": 1, "TRANSFORMERS_OFFLINE": 1, "WANDB_MODE": "offline",
        "VERL_STEP_CACHE_DIR": run / "step_cache", "NCCL_DEBUG": "INFO",
    }
    for name in list(os.environ):
        if name.startswith("SP_Q_") or name.startswith("SP_REPLAY_"):
            del os.environ[name]
    if args.method != "grpo":
        seed = run / "cold"
        if not seed.exists():
            subprocess.run([sys.executable, str(ROOT / "experiments/08_13_tiedq_seed192/build_cold_artifacts.py"),
                            "--out", str(seed)], check=True)
        env.update({"SP_REPLAY_ENABLE": 1, "SP_REPLAY_COLD_BOOTSTRAP": 1,
                    "SP_REPLAY_SEED_DIR": seed / "replay_seed_cold",
                    "SP_REPLAY_N": batch, "SP_SCRATCH_INFLOW_ONLY": 1,
                    "SP_REPLAY_BUCKETING": "global", "SP_REPLAY_ADMISSION": "ungated",
                    "SP_REPLAY_ROTATION": "global_fifo", "SP_REPLAY_BOUND": args.replay_bound,
                    "SP_REPLAY_GLOBAL_SAMPLING": "question", "SP_REPLAY_CUT_LOW": 0,
                    "SP_REPLAY_CUT_HIGH": .9, "SP_REPLAY_CUT_GRAIN": args.chunk,
                    "SP_REPLAY_POLICY_OVERRIDE": 0})
        if args.method == "ac2":
            env.update({"SP_Q_ENABLE": 1, "SP_Q_SEPARATE": 0,
                        "SP_Q_SEED_DIR": seed, "SP_Q_BANK_DIR": seed,
                        "SP_Q_INTERLEAVE": 1, "SP_Q_INTERLEAVE_AFTER": 1,
                        "SP_Q_LR_LADDER": 1, "SP_Q_LR_INITIAL": "2.8284271247461903e-6",
                        "SP_Q_LR_FLOOR": "7.071067811865476e-7",
                        "SP_Q_LR_RATIO_MAX": "1.4142135623730951",
                        "SP_Q_LR_BREACH_PATIENCE": 2, "SP_Q_LR_REDUCTION_FACTOR": .5,
                        "SP_Q_BUDGET_G": args.chunk, "SP_Q_CTX_LIMIT": args.response + 3296,
                        "SP_Q_MAX_TOKEN_LEN": args.response + 3360,
                        "SP_Q_READY_THRESH_GLOBAL": .2, "SP_Q_READY_THRESH_PROBLEM": .18,
                        "SP_Q_READY_REQUIRE_BANK": 1, "SP_Q_REQUIRE_NONZERO": 1,
                        "SP_Q_FIFO_CAP": 1920, "SP_Q_TRAIN_N": args.q_train_n if args.q_train_n is not None else (32 if args.smoke else 768),
                        "SP_Q_MIN_VALID": min(8, args.group), "SP_Q_GRAD_CLIP": .2,
                        "SP_Q_TRAIN_NOREF": 1, "SP_Q_PROMPT_VARIANT": "reward_horizon",
                        "SP_Q_REF_REQUIRE_PASS": 1, "SP_Q_AUDIT_DEN": 4,
                        "SP_Q_AUDIT_CUT": 1, "SP_Q_DUMP_WAVE": 1})
        else:
            env["SP_Q_ENABLE"] = 0
    else:
        env.update({"SP_REPLAY_ENABLE": 0, "SP_Q_ENABLE": 0, "SP_SCRATCH_INFLOW_ONLY": 0})
    if args.ablation != "none":
        assert args.method == "ac2"
        if args.ablation == "correct-only":
            env.update({"SP_REPLAY_ADMISSION": "judged_correct", "SP_PASS_POINTS_MIN": 6,
                        "SP_Q_RNG_SEED": 826001})
        elif args.ablation == "no-audit":
            env.update({"SP_Q_AUDIT_DEN": 0, "SP_Q_AUDIT_CUT": 0, "SP_Q_RNG_SEED": 831001})
    if args.engineering_readiness:
        assert args.smoke and args.method == "ac2", "permissive readiness is engineering-only"
        env.update({"SP_Q_READY_THRESH_GLOBAL": 1.01, "SP_Q_READY_THRESH_PROBLEM": 1.01,
                    "SP_Q_READY_REQUIRE_BANK": 0, "SP_Q_REQUIRE_NONZERO": 0})
        print("ENGINEERING_READINESS_FIXTURE: permissive thresholds; no scientific readiness claim", flush=True)
    os.environ.update({k: str(v) for k, v in env.items()})
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=EXPERIMENTS, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gpus", type=int, default=7)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--save-freq", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--group", type=int, default=2)
    p.add_argument("--response", type=int, default=256)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--data", default="runs/e3/canonical")
    p.add_argument("--val-n", type=int, default=4)
    p.add_argument("--val-freq", type=int, default=10)
    p.add_argument("--judge-max-tokens", type=int, choices=[40000, 65536], default=40000)
    p.add_argument("--replay-bound", type=int, default=256)
    p.add_argument("--q-train-n", type=int)
    p.add_argument("--ablation", choices=["none", "correct-only", "no-audit"], default="none")
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--initial-val", action="store_true",
                   help="validate before training; R cold launches use this, resumes do not")
    p.add_argument("--strict-judge", action="store_true",
                   help="abort on judge failures instead of training on failed zero rewards")
    p.add_argument("--val-data", type=Path,
                   help="explicit validation parquet (bounded R smoke uses fixed problem0)")
    p.add_argument("--engineering-readiness", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compose-only", action="store_true")
    args = p.parse_args()
    run = configure(args)
    exp = ROOT / "experiments" / EXPERIMENTS[args.method]
    sys.path.insert(0, str(exp))
    spec = importlib.util.spec_from_file_location("paper_runner", exp / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = runner.build_config()
    from omegaconf import OmegaConf, open_dict
    with open_dict(cfg):
        if args.strict_judge:
            cfg.reward.custom_reward_function.path = str(ROOT / "repro/strict_reward.py")
        if args.val_data is not None:
            val_data = args.val_data.resolve()
            assert val_data.is_relative_to(ROOT / "runs") and val_data.is_file()
            cfg.data.val_files = [str(val_data)]
        cfg.trainer.logger = ["console", "file"]
        cfg.reward.reward_model.enable_resource_pool = False
        cfg.actor_rollout_ref.rollout.enforce_eager = True
        cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config = json.dumps({"mode": 0, "cudagraph_mode": "NONE"})
        cfg.actor_rollout_ref.rollout.agent.num_workers = 4
        cfg.ray_kwargs.ray_init.num_cpus = 24
        cfg.ray_kwargs.ray_init.num_gpus = args.gpus
        cfg.ray_kwargs.ray_init.include_dashboard = False
        cfg.ray_kwargs.ray_init.object_store_memory = 2 * 1024**3
        cfg.ray_kwargs.ray_init._temp_dir = os.environ["RAY_TMPDIR"]
        # Ray assigns CUDA_VISIBLE_DEVICES per actor; never override that assignment.
        forwarded = cfg.ray_kwargs.ray_init.runtime_env.env_vars
        for k, v in os.environ.items():
            if k.startswith(("SP_", "HF_")) or k in (
                "TMPDIR", "TMP", "TEMP", "RAY_TMPDIR", "UV_CACHE_DIR", "TIKTOKEN_CACHE_DIR",
                "SELF_PLAY_JUDGE_URL", "FLASHINFER_WORKSPACE_BASE", "NCCL_DEBUG"):
                forwarded[k] = v
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest = run / "launches" / stamp
    manifest.mkdir(parents=True)
    OmegaConf.save(cfg, manifest / "config.yaml")
    (manifest / "arguments.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"COMPOSE_OK method={args.method} world_size={args.gpus} rollout_tp={args.tp} rollout_dp={args.gpus // args.tp} manifest={manifest}", flush=True)
    assert cfg.reward.reward_model.enable is False
    if args.compose_only:
        return
    from ac2.utils.experiment_utils import manifest_dump
    manifest_dump(manifest / "source", cfg, src_dir=ROOT / "src", extra_files=[
        Path(__file__), exp / "runner.py", ROOT / "repro/strict_reward.py"])
    metrics_path = run / "metrics.jsonl"
    before = metrics_path.read_bytes() if metrics_path.exists() else b""
    (manifest / "metrics_before.jsonl").write_bytes(before)
    (manifest / "judge_before.json").write_text(json.dumps(judge_accounting(), indent=2))
    runner.main_ppo.run_ppo(cfg)
    after = metrics_path.read_bytes()
    assert after.startswith(before), "metrics history was rewritten during launch"
    (manifest / "metrics_added.jsonl").write_bytes(after[len(before):])
    (manifest / "judge_after.json").write_text(json.dumps(judge_accounting(), indent=2))
    print(f"RUN_COMPLETED metrics_added={manifest / 'metrics_added.jsonl'}", flush=True)
    import ray
    ray.shutdown()


if __name__ == "__main__":
    main()
