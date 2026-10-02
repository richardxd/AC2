"""Local, stage-resumable R5 pipeline; enclose each smoke stage in run_bounded.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/08_15_q_probe_step40"
STAGES = ["prepare", "generate", "critic", "judge", "analyze"]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(cmd, log, env=None):
    with log.open("x") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES + ["verify-resume"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--shards", type=int, default=7)
    p.add_argument("--verify-step", type=int, required=True)
    p.add_argument("--verify-chunk", type=int, default=4096)
    p.add_argument("--engineering-fixture", action="store_true")
    args = p.parse_args()
    for key in ("run", "data", "out"):
        value = getattr(args, key).resolve()
        assert value.is_relative_to(ROOT / "runs")
        setattr(args, key, value)
    assert 1 <= args.shards <= 7 and args.groups >= args.shards
    assert 0 < args.verify_step <= args.step
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / f"checkpoints/global_step_{args.step}"
    source_launches = sorted((args.run / "launches").glob("*/arguments.json"))
    assert source_launches
    source_args = [json.loads(path.read_text()) for path in source_launches]
    if not args.engineering_fixture:
        assert all(not a["engineering_readiness"] for a in source_args), "fixture requires explicit label"
        assert all(a["method"] == "ac2" and a["response"] == 16384 and a["chunk"] == 4096
                   for a in source_args), "scientific source does not match R2 protocol"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "stage"}
    config.update(response=16384, chunk=4096, samples=4, q_temperature=0,
                  policy_temperature=.8, top_p=1, top_k=-1, prefix_seed=192)
    configuration = args.out / "configuration.json"
    if configuration.exists():
        assert json.loads(configuration.read_text()) == config, "resume configuration changed"
    else:
        with configuration.open("x") as f:
            json.dump(config, f, indent=2)
    done = {}
    for stage in STAGES:
        receipt = args.out / f"{stage}.done.json"
        if receipt.exists():
            done[stage] = json.loads(receipt.read_text())
            for path, sha in done[stage]["sha256"].items():
                assert digest(Path(path)) == sha, f"resume artifact changed: {path}"
    if args.stage == "verify-resume":
        assert set(done) == set(STAGES), "pipeline incomplete"
        print("RESUME_VERIFIED all five stages; no model generation or judge calls repeated")
        return
    if args.stage in done:
        print(f"RESUME_VERIFIED {args.stage}; preserved completed outputs")
        return
    assert all(s in done for s in STAGES[:STAGES.index(args.stage)]), "prior stage incomplete"
    started = time.monotonic()
    model = args.out / "model_hf"
    common_q = ["--run-dir", str(args.run), "--model", str(model), "--bank-dir", str(args.run / "cold")]
    outputs = []
    if args.stage == "prepare":
        assert checkpoint.is_dir() and not model.exists()
        run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
             "--local_dir", str(checkpoint / "actor"), "--target_dir", str(model), "--use_cpu_initialization"],
            args.out / "merge.log")
        run([sys.executable, "-m", "verl.model_merger", "test", "--backend", "fsdp",
             "--local_dir", str(checkpoint / "actor"), "--test_hf_dir", str(model), "--use_cpu_initialization"],
            args.out / "merge-verify.log")
        # Every inference call explicitly requests BF16; preserve the merger's config.
        run([sys.executable, str(EXP / "probe_q.py"), "--verify", *common_q,
             "--verify-step", str(args.verify_step), "--budget-g", str(args.verify_chunk)], args.out / "verify.log")
        run([sys.executable, str(EXP / "build_probe_set.py"), "--run-dir", str(args.run),
             "--step", str(args.step), "--data-dir", str(args.data), "--n", str(args.groups),
             "--resp-cap", "16384", "--budget-g", "4096", "--min-budget", "6144",
             "--out", str(args.out / "probe_set.jsonl")], args.out / "selection.log")
        rows = [json.loads(x) for x in (args.out / "probe_set.jsonl").read_text().splitlines()]
        assert len(rows) == args.groups and len({r["qid"] for r in rows}) == args.groups
        outputs = list(model.glob("*")) + [args.out / "probe_set.jsonl", checkpoint / "q_state.json"]
        outputs += sorted((checkpoint / "actor").glob("model_world_size_*_rank_*.pt"))
        outputs += [args.data / "train.parquet", args.run / "replay_buffer_deltas.jsonl",
                    args.run / "q_state_deltas.jsonl", args.out / "merge-verify.log", args.out / "verify.log"]
    elif args.stage in {"generate", "critic"}:
        folder = args.out / ("gen" if args.stage == "generate" else "q")
        folder.mkdir(exist_ok=False)
        children = []
        for shard in range(args.shards):
            output = folder / f"{'gen' if args.stage == 'generate' else 'q'}.shard{shard}.jsonl"
            cmd = [sys.executable, str(EXP / ("probe_gen.py" if args.stage == "generate" else "probe_q.py")),
                   "--shard", str(shard), "--num-shards", str(args.shards), "--out", str(output),
                   "--tp", "1", "--gpu-mem-util", ".75", "--max-num-seqs", "4", "--enforce-eager", "--budget-g", "4096"]
            if args.stage == "generate":
                cmd += ["--model", str(model), "--probe-set", str(args.out / "probe_set.jsonl"),
                        "--n", "4", "--keep-ids", "16384", "--max-model-len", "18432"]
            else:
                cmd += common_q + ["--gen-dir", str(args.out / "gen"), "--upto-step", str(args.step-1),
                        "--temperature", "0", "--max-model-len", "19680"]
            log = (args.out / f"{args.stage}-{shard}.log").open("x")
            children.append((subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard+1)),
                                             stdout=log, stderr=subprocess.STDOUT), log))
            outputs.append(output)
        codes = []
        for child, log in children:
            codes.append(child.wait())
            log.close()
        assert codes == [0] * args.shards, codes
    elif args.stage == "judge":
        from local_runner import judge_accounting
        before = judge_accounting()
        with (args.out / "judge_before.json").open("x") as f:
            json.dump(before, f)
        folder = args.out / "judged"
        folder.mkdir(exist_ok=False)
        output = folder / "judged.shard0.jsonl"
        run([sys.executable, str(EXP / "probe_judge.py"), "--gen-dir", str(args.out / "gen"),
             "--data-dir", str(args.data), "--out", str(output), "--concurrency", "4",
             "--judge-url", "http://127.0.0.1:18791/v1"], args.out / "judge.log",
            dict(os.environ, SP_JUDGE_MAX_INFLIGHT="4"))
        with (args.out / "judge_after.json").open("x") as f:
            json.dump(judge_accounting(), f)
        outputs = [output, args.out / "judge_before.json", args.out / "judge_after.json"]
    else:
        output = args.out / "report.json"
        run([sys.executable, str(ROOT / "repro/probe_report.py"), "--directory", str(args.out),
             "--n", "4", "--output", str(output)], args.out / "analysis.log")
        outputs = [output]
    assert outputs and all(path.is_file() for path in outputs)
    receipt = {"stage": args.stage, "elapsed_s": time.monotonic()-started,
               "engineering_fixture": args.engineering_fixture,
               "sha256": {str(path): digest(path) for path in outputs + [configuration]}}
    with (args.out / f"{args.stage}.done.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({k: receipt[k] for k in ("stage", "elapsed_s", "engineering_fixture")}))


if __name__ == "__main__":
    main()
