"""GRPO baseline training entry point: base model, no replay buffer, no critic.

Builds the verl PPO/Hydra configuration from SP_* environment variables, writes the run
manifest and calls verl.trainer.main_ppo.run_ppo. Starting from the base
Qwen/Qwen3-4B-Thinking-2507 model with a fresh optimizer, every step samples
SP_TRAIN_BATCH_SIZE fresh problems from a plain RLHFDataset (standard shuffle) and
SP_ROLLOUT_N responses per problem, rolls each response out to completion and trains on
all of them. The advantage is the group-mean-centred reward (GRPO without std
normalisation) and there is no KL term. The training reward is the fine-grained
no-reference judge (reward = points/7, row_pass = points >= SP_PASS_POINTS_MIN);
validation uses the IMO-ProofBench ProofAutoGrader template.

This file is identical in 07_15_handoff, 07_15_handoff_lr1e6 and 07_15_handoff_lr4e6. The
values below are fallbacks; run_attach_cluster_c.sh pins the configuration of each arm
(including SP_LR).
"""

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

import verl.trainer.main_ppo as main_ppo
import ac2.rewards.ds4_finegrained_judge as ds4_judge_reward
from ac2.utils.experiment_utils import manifest_dump

try:
    from ac2.clusters import cluster_config
    _CLUSTER = cluster_config()
except Exception:
    _CLUSTER = {
        "hf_hub_cache": os.path.expandvars("${AC2_CLUSTER_B_SCRATCH}/.cache/huggingface/hub"),
        "hf_home": os.path.expanduser("~/.cache/huggingface"),
        "cache_root": os.path.expandvars("${AC2_CLUSTER_B_ROOT}/.cache"),
        "wandb_mode": "offline",
    }

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = os.environ.get("SP_EXPERIMENT_NAME", EXP_DIR.name)
RUN_DATA_NAME = os.environ.get("SP_RUN_DATA_DIR", "run_data")
REPO_ROOT = EXP_DIR.parents[1]
MANIFEST_DIR = EXP_DIR / "manifest"
RUN_DATA = EXP_DIR / RUN_DATA_NAME
METRICS_PATH = RUN_DATA / "metrics.jsonl"
WANDB_ENTITY = os.environ.get("AC2_WANDB_ENTITY", "")

# DS4 fine-grained judge: reward = points/7; row_pass = points >= 6.
REWARD_FILE = ds4_judge_reward.__file__
FINEGRAINED_TEMPLATE = str(Path(REWARD_FILE).parent / "templates" / "finegrained_noref_judge.txt")
VAL_JUDGE_TEMPLATE = str(Path(REWARD_FILE).parent / "templates" / "imo_proofautograder.txt")
VAL_MAP_PATH = os.path.expanduser(
    os.environ.get("SP_VAL_MAP", "~/data/fineproofs/val_map.json"))
VERL_CONFIG_DIR = str(Path(main_ppo.__file__).parent / "config")

NNODES = int(os.environ.get("NNODES", 4))
N_GPUS_PER_NODE = int(os.environ.get("N_GPUS_PER_NODE", 8))
DATA_DIR = os.environ.get("SELF_PLAY_DATA_DIR", f"{os.environ['HOME']}/data/fineproofs")


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default



# A `<run_data>/.target_total_steps` file, if present, OVERRIDES SP_TOTAL_STEPS so the
# target can be raised for the NEXT resume without an interrupt/resubmit.
_target_file = RUN_DATA / ".target_total_steps"
TOTAL_STEPS = (int(_target_file.read_text().strip()) if _target_file.exists()
               else _env("SP_TOTAL_STEPS", 100, int))
TRAIN_BATCH       = _env("SP_TRAIN_BATCH_SIZE", 256, int)   # problems per step
ROLLOUT_N         = _env("SP_ROLLOUT_N", 16, int)
PPO_MINI          = min(_env("SP_PPO_MINI_BATCH", 128, int), TRAIN_BATCH)
LR                = _env("SP_LR", 1.4142135623730951e-6, float)  # fallback (1e-6 * sqrt2); the launch script sets SP_LR
MAX_PROMPT_LEN    = _env("SP_MAX_PROMPT_LEN", 2048, int)
MAX_RESPONSE_LEN  = _env("SP_MAX_RESPONSE_LEN", 50000, int)
ROLLOUT_MAXLEN    = MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1248
VAL_N             = _env("SP_VAL_N", 16, int)
TEST_FREQ         = _env("SP_TEST_FREQ", 5, int)
SAVE_FREQ         = _env("SP_SAVE_FREQ", 1, int)
VAL_BEFORE_TRAIN  = _env("SP_VAL_BEFORE_TRAIN", "True")
VAL_ONLY          = _env("SP_VAL_ONLY", "False")



def _latest_ckpt_step() -> int:
    f = RUN_DATA / "checkpoints" / "latest_checkpointed_iteration.txt"
    try:
        return int(f.read_text().strip())
    except (OSError, ValueError):
        return 0


_resume_step = _latest_ckpt_step()
print(f"[resume] latest ckpt step = {_resume_step}; target = {TOTAL_STEPS}; "
      f"train_batch={TRAIN_BATCH} ppo_mini={PPO_MINI} lr={LR:.6g} exp_name={EXP_NAME}", flush=True)
ACTOR_MEM_UTIL    = _env("SP_ACTOR_GPU_MEM_UTIL", 0.6, float)
ACTOR_MODEL       = _env("SP_ACTOR_MODEL", "Qwen/Qwen3-4B-Thinking-2507")

# ---- judge (reward_model) knobs: DeepSeek-V4-Flash (DS4) judge ----
_DS4_SNAPSHOT_DEFAULT = (
    os.path.expandvars("${AC2_CLUSTER_B_SCRATCH}/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1")
)
JUDGE_MODEL       = _env("SP_JUDGE_MODEL", _DS4_SNAPSHOT_DEFAULT)
JUDGE_TP          = _env("SP_JUDGE_TP", 8, int)
JUDGE_MAXLEN      = _env("SP_JUDGE_MAXLEN", 98304, int)
JUDGE_RESP_LEN    = _env("SP_JUDGE_RESP_LEN", 40960, int)
JUDGE_MAX_TOKENS  = _env("SP_JUDGE_MAX_TOKENS", 40000, int)
JUDGE_REASONING   = _env("SP_JUDGE_REASONING", "high")
JUDGE_MEM_UTIL    = _env("SP_JUDGE_GPU_MEM_UTIL", 0.80, float)
JUDGE_ENFORCE_EAGER = _env("SP_JUDGE_ENFORCE_EAGER", "False")
JUDGE_STANDALONE  = _env("SP_JUDGE_STANDALONE", "False") in ("1", "True", "true")
JUDGE_CUDAGRAPH_MODE = _env("SP_JUDGE_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
PASS_POINTS_MIN   = _env("SP_PASS_POINTS_MIN", 6, int)
PPO_MAX_TOKEN_LEN = _env("SP_PPO_MAX_TOKEN_LEN", _CLUSTER.get("ppo_max_token_len", 51200), int)
ENTROPY_COEFF     = _env("SP_ENTROPY_COEFF", 0.0, float)
LOG_PROB_MAX_TOKEN_LEN = _env("SP_LOG_PROB_MAX_TOKEN_LEN", 3 * PPO_MAX_TOKEN_LEN, int)
USE_FUSED_KERNELS = _env("SP_USE_FUSED_KERNELS", "True")
ROLLOUT_TP        = _env("SP_ROLLOUT_TP", 1, int)
ACTOR_MAX_NUM_SEQS = _env("SP_ACTOR_MAX_NUM_SEQS", _CLUSTER.get("rollout_max_num_seqs", 96), int)
MAX_NUM_BATCHED_TOKENS = _env("SP_MAX_NUM_BATCHED_TOKENS", 32768, int)
REWARD_MAX_NUM_SEQS = _env("SP_REWARD_MAX_NUM_SEQS", 256, int)
REWARD_MAX_NUM_BATCHED_TOKENS = _env("SP_REWARD_MAX_NUM_BATCHED_TOKENS", 16384, int)
TEMPERATURE       = _env("SP_TEMPERATURE", 0.8, float)
VAL_TOP_P         = _env("SP_VAL_TOP_P", 0.95, float)
VAL_TOP_K         = _env("SP_VAL_TOP_K", 20, int)
MAX_CKPT_KEEP     = _env("SP_MAX_CKPT_KEEP", 5, int)
REWARD_NUM_WORKERS = _env("SP_REWARD_NUM_WORKERS", 8, int)

OVERRIDES = [
    # ---- GRPO, no KL ----
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "algorithm.norm_adv_by_std_in_grpo=False",
    # ---- reward = async DS4 fine-grained judge ----
    f"reward.custom_reward_function.path={REWARD_FILE}",
    "reward.custom_reward_function.name=compute_score",
    f"+reward.custom_reward_function.reward_kwargs.judge_max_tokens={JUDGE_MAX_TOKENS}",
    f"+reward.custom_reward_function.reward_kwargs.judge_reasoning_effort={JUDGE_REASONING}",
    "+reward.custom_reward_function.reward_kwargs.judge_temperature=1.0",
    "+reward.custom_reward_function.reward_kwargs.judge_top_p=1.0",
    "+reward.custom_reward_function.reward_kwargs.judge_top_k=-1",
    "+reward.custom_reward_function.reward_kwargs.judge_seed=42",
    "+reward.custom_reward_function.reward_kwargs.judge_payload_style=deepseek_v4",
    f"+reward.custom_reward_function.reward_kwargs.pass_points_min={PASS_POINTS_MIN}",
    f"+reward.custom_reward_function.reward_kwargs.finegrained_template_path={FINEGRAINED_TEMPLATE}",
    f"+reward.custom_reward_function.reward_kwargs.val_judge_template_path={VAL_JUDGE_TEMPLATE}",
    f"+reward.custom_reward_function.reward_kwargs.val_map_path={VAL_MAP_PATH}",
    "reward.reward_manager.name=naive",
    f"reward.num_workers={REWARD_NUM_WORKERS}",
    # ---- colocated DS4-Flash judge ----
    f"reward.reward_model.enable={not bool(os.environ.get('SELF_PLAY_JUDGE_URL'))}",
    "reward.reward_model.enable_resource_pool={}".format(JUDGE_STANDALONE),
    f"reward.reward_model.model_path={JUDGE_MODEL}",
    "reward.reward_model.rollout.name=vllm",
    f"reward.reward_model.rollout.tensor_model_parallel_size={JUDGE_TP}",
    f"reward.reward_model.rollout.gpu_memory_utilization={JUDGE_MEM_UTIL}",
    f"reward.reward_model.rollout.max_num_seqs={REWARD_MAX_NUM_SEQS}",
    f"reward.reward_model.rollout.max_num_batched_tokens={REWARD_MAX_NUM_BATCHED_TOKENS}",
    f"reward.reward_model.rollout.max_model_len={JUDGE_MAXLEN}",
    f"reward.reward_model.rollout.response_length={JUDGE_RESP_LEN}",
    "reward.reward_model.rollout.free_cache_engine=True",
    f"reward.reward_model.rollout.enforce_eager={JUDGE_ENFORCE_EAGER}",
    "reward.reward_model.rollout.enable_chunked_prefill=True",
    "reward.reward_model.rollout.enable_prefix_caching=True",
    # ---- DS4 engine args ----
    "+reward.reward_model.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4",
    "+reward.reward_model.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4",
    "+reward.reward_model.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8",
    "+reward.reward_model.rollout.engine_kwargs.vllm.block_size=256",
    "+reward.reward_model.rollout.engine_kwargs.vllm.moe_backend=marlin",
    "+reward.reward_model.rollout.engine_kwargs.vllm.async_scheduling=True",
    # compilation_config injected post-compose as a JSON STRING in build_config().
    # ---- data (prover-only; plain RLHFDataset — no replay) ----
    f"data.train_files={DATA_DIR}/train.parquet",
    f"data.val_files={DATA_DIR}/test.parquet",
    f"data.train_batch_size={TRAIN_BATCH}",
    f"data.max_prompt_length={MAX_PROMPT_LEN}",
    f"data.max_response_length={MAX_RESPONSE_LEN}",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    # ---- model ----
    f"actor_rollout_ref.model.path={ACTOR_MODEL}",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    # ---- actor ----
    f"actor_rollout_ref.actor.optim.lr={LR}",
    "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
    "actor_rollout_ref.actor.optim.weight_decay=0.01",
    f"actor_rollout_ref.actor.ppo_mini_batch_size={PPO_MINI}",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.kl_loss_coef=0.0",
    f"actor_rollout_ref.actor.entropy_coeff={ENTROPY_COEFF}",
    "actor_rollout_ref.actor.grad_clip=0.3",
    "actor_rollout_ref.actor.clip_ratio=0.2",
    "actor_rollout_ref.actor.clip_ratio_low=0.2",
    "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.actor.clip_ratio_c=3.0",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={PPO_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={LOG_PROB_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={LOG_PROB_MAX_TOKEN_LEN}",
    f"actor_rollout_ref.model.use_fused_kernels={USE_FUSED_KERNELS}",
    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    # ---- rollout (vLLM); n = GRPO group size ----
    "actor_rollout_ref.rollout.name=vllm",
    f"actor_rollout_ref.rollout.n={ROLLOUT_N}",
    f"actor_rollout_ref.rollout.max_model_len={ROLLOUT_MAXLEN}",
    f"actor_rollout_ref.rollout.temperature={TEMPERATURE}",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.top_k=-1",
    f"actor_rollout_ref.rollout.tensor_model_parallel_size={ROLLOUT_TP}",
    f"actor_rollout_ref.rollout.max_num_seqs={ACTOR_MAX_NUM_SEQS}",
    f"actor_rollout_ref.rollout.max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS}",
    f"actor_rollout_ref.rollout.gpu_memory_utilization={ACTOR_MEM_UTIL}",
    "actor_rollout_ref.rollout.enable_chunked_prefill=True",
    "actor_rollout_ref.rollout.calculate_log_probs=True",
    f"actor_rollout_ref.rollout.val_kwargs.n={VAL_N}",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    f"actor_rollout_ref.rollout.val_kwargs.temperature={TEMPERATURE}",
    f"actor_rollout_ref.rollout.val_kwargs.top_p={VAL_TOP_P}",
    f"actor_rollout_ref.rollout.val_kwargs.top_k={VAL_TOP_K}",
    # ---- trainer / logging ----
    "trainer.logger=[console,wandb,file]",
    "trainer.project_name=self-play",
    f"trainer.experiment_name={EXP_NAME}",
    f"trainer.default_local_dir={RUN_DATA / 'checkpoints'}",
    f"trainer.rollout_data_dir={RUN_DATA / 'rollouts'}",
    f"trainer.validation_data_dir={RUN_DATA / 'val_rollouts'}",
    "trainer.log_val_generations=16",
    f"trainer.n_gpus_per_node={N_GPUS_PER_NODE}",
    f"trainer.nnodes={NNODES}",
    f"trainer.val_before_train={VAL_BEFORE_TRAIN}",
    f"trainer.val_only={VAL_ONLY}",
    f"trainer.test_freq={TEST_FREQ}",
    f"trainer.save_freq={SAVE_FREQ}",
    f"trainer.max_actor_ckpt_to_keep={MAX_CKPT_KEEP}",
    "trainer.total_epochs=9999",
    f"trainer.total_training_steps={TOTAL_STEPS}",
    "trainer.resume_mode=auto",
]
if JUDGE_STANDALONE:
    OVERRIDES += [
        "reward.reward_model.nnodes=1",
        f"reward.reward_model.n_gpus_per_node={JUDGE_TP}",
    ]

if os.environ.get("SELF_PLAY_JUDGE_URL"):
    OVERRIDES.append(f"+reward.custom_reward_function.reward_kwargs.judge_url={os.environ['SELF_PLAY_JUDGE_URL']}")
# One-time optimizer reset (pass SP_RESET_OPTIMIZER=1 for a SINGLE re-attach only).
# Not needed for an ordinary resume: with the same world size the full optimizer state resumes.
if _env("SP_RESET_OPTIMIZER", 0, int):
    OVERRIDES.append("actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]")
    print(f"[ckpt] SP_RESET_OPTIMIZER=1 -> FRESH optimizer on this resume (from ckpt {_resume_step})",
          flush=True)

# CUDA-graph configuration of the judge engine (injected as a JSON string in build_config()).
_JUDGE_COMPILATION_CONFIG = json.dumps({
    "mode": 0,
    "cudagraph_mode": JUDGE_CUDAGRAPH_MODE,
    "cudagraph_capture_sizes": [
        1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
        120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224,
        232, 240, 248, 256,
    ],
})


def _engine_cache_env():
    """Cache/telemetry env for all Ray actors."""
    xdg = os.environ.get("XDG_CACHE_HOME", _CLUSTER.get("cache_root", os.path.expandvars("${AC2_CLUSTER_B_ROOT}/.cache")))
    d = {
        "XDG_CACHE_HOME": xdg,
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", f"{xdg}/triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"{xdg}/inductor"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{xdg}/vllm"),
        "TORCH_HOME": os.environ.get("TORCH_HOME", f"{xdg}/torch"),
        "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH", f"{xdg}/nv"),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR", f"{xdg}/torch_extensions"),
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", f"{xdg}/matplotlib"),
        "WANDB_CACHE_DIR": os.environ.get("WANDB_CACHE_DIR", f"{xdg}/wandb"),
        "WANDB_DATA_DIR": os.environ.get("WANDB_DATA_DIR", f"{xdg}/wandb-data"),
        "WANDB_MODE": os.environ.get("WANDB_MODE", _CLUSTER.get("wandb_mode", "offline")),
        "FLASHINFER_WORKSPACE_BASE": os.environ.get(
            "FLASHINFER_WORKSPACE_BASE", os.path.dirname(xdg)
        ),
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get("VLLM_DISABLE_COMPILE_CACHE", "1"),
        "VLLM_DEEP_GEMM_WARMUP": os.environ.get("VLLM_DEEP_GEMM_WARMUP", "skip"),
        "VLLM_ENGINE_READY_TIMEOUT_S": os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "3600"),
        "VLLM_RPC_TIMEOUT": os.environ.get("VLLM_RPC_TIMEOUT", "600000"),
        "TILELANG_CLEANUP_TEMP_FILES": os.environ.get("TILELANG_CLEANUP_TEMP_FILES", "1"),
        "TILELANG_CACHE_DIR": os.environ.get("TILELANG_CACHE_DIR", f"{xdg}/tilelang"),
        "TILELANG_TMP_DIR": os.environ.get("TILELANG_TMP_DIR", f"{xdg}/tilelang/tmp"),
        "TRTLLM_DG_CACHE_DIR": os.environ.get("TRTLLM_DG_CACHE_DIR", f"{xdg}/trtllm_dg"),
        # PER-NODE deep_gemm JIT cache: first-use compiles into a cache shared across nodes can
        # race (a node may load another node's half-written cubin). The per-host suffix is
        # resolved in vllm_async_server.py.
        "SP_DG_JIT_CACHE_BASE": os.environ.get("SP_DG_JIT_CACHE_BASE", f"{xdg}/dg_jit_nodes"),
        "DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
    }
    return d


def build_config():
    with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=OVERRIDES)
    main_ppo.auto_set_device(cfg)
    cfg = main_ppo.migrate_legacy_reward_impl(cfg)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE", _CLUSTER.get("hf_hub_cache", ""))
    hf_home = os.environ.get("HF_HOME", _CLUSTER.get("hf_home", os.path.expanduser("~/.cache/huggingface")))
    env_vars = {
        "VERL_FILE_LOGGER_PATH": str(METRICS_PATH),
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_RUN_ID": EXP_NAME,
        "WANDB_RESUME": os.environ.get("WANDB_RESUME", "allow"),
        "HF_HOME": hf_home,
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        **_engine_cache_env(),
    }
    if os.environ.get("WANDB_MODE", "online") != "offline":
        try:
            import netrc as _netrc_mod
            _wb = _netrc_mod.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
            if _wb and _wb[2]:
                env_vars["WANDB_API_KEY"] = _wb[2]
        except Exception as _e:
            print(f"[wandb] could not read ~/.netrc api.wandb.ai key ({_e}); relying on actor env")
    if hf_hub_cache:
        env_vars["HF_HUB_CACHE"] = hf_hub_cache
    _step_cache = os.environ.get("VERL_STEP_CACHE_DIR")
    if _step_cache:
        env_vars["VERL_STEP_CACHE_DIR"] = _step_cache
    for _k in (
        "SP_JUDGE_HTTP_TOTAL_TIMEOUT", "SP_JUDGE_HTTP_CONN_LIMIT", "SP_JUDGE_MAX_INFLIGHT",
        "SP_LENPEN_ENABLE", "SP_LENPEN_START", "SP_LENPEN_END", "SP_LENPEN_MAX", "SP_ROUTER_HTTP_TOTAL_TIMEOUT",
        "VERL_STAGGER_ENGINE_INIT",
        "SP_PASS_POINTS_MIN",
        "SP_ADAPTIVE_ENTROPY", "SP_AEC_TARGET_H", "SP_AEC_DELTA",
        "SP_AEC_KMAX", "SP_AEC_KMIN", "SP_AEC_KINIT",
        "SP_DIFF_SAMPLING", "SP_DIFF_WEIGHT", "SP_DIFF_W_FLOOR", "SP_DIFF_EMA_ALPHA",
        "SP_DIFF_MIN_OBS", "SP_DIFF_P_PRIOR", "SP_DIFF_PASS_THRESH", "SP_DIFF_THRESH",
        "SP_DIFF_HYST_LOW", "SP_DIFF_W_LOW", "SP_DIFF_WARMSTART",
        "SP_KEEP_BEST_CKPT",
        # Retention stride (trainer default 10 = keep every 10th step's weights forever).
        # Only the OFF-switch needs forwarding: unset means the actor falls back to the
        # trainer default, which is the intended behaviour. Forwarded so SP_CKPT_KEEP_EVERY=0
        # actually reaches the driver -- the forward loop carries "0" (a non-empty string).
        "SP_CKPT_KEEP_EVERY",
        "SP_ROLLOUT_BACKFILL", "SP_ROLLOUT_BACKFILL_DIM",
    ):
        _v = os.environ.get(_k)
        if _v:
            env_vars[_k] = _v
    for _k in (
        "CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CUDACXX",
        "TRTLLM_DG_NVCC_COMPILER", "CPATH", "NVCC_PREPEND_FLAGS",
    ):
        _v = os.environ.get(_k)
        if _v:
            env_vars[_k] = _v
    with open_dict(cfg):
        cfg.ray_kwargs.ray_init.runtime_env = {"env_vars": env_vars}
        cfg.reward.reward_model.rollout.engine_kwargs.vllm.compilation_config = _JUDGE_COMPILATION_CONFIG
        if os.environ.get("SP_RAY_LOCAL", "0") in ("1", "true", "True"):
            cfg.ray_kwargs.ray_init.address = "local"
            print("[ray] SP_RAY_LOCAL=1 -> ray.init(address='local')", flush=True)
    return cfg


def main():
    cfg = build_config()
    manifest_dump(
        MANIFEST_DIR,
        cfg,
        src_dir=REPO_ROOT / "src",
        extra_files=[Path(__file__).resolve(), EXP_DIR / "run_attach_cluster_c.sh"],
    )
    RUN_DATA.mkdir(parents=True, exist_ok=True)
    for _cache_dir in _engine_cache_env().values():
        if isinstance(_cache_dir, str) and _cache_dir.startswith("/"):
            Path(_cache_dir).mkdir(parents=True, exist_ok=True)
    main_ppo.run_ppo(cfg)


if __name__ == "__main__":
    main()
