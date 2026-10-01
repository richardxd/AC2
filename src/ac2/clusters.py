"""Per-cluster configuration registry.

Site-specific paths, accounts and the detection markers come from the environment (the
repository's .env; see .env.example). Keeps cluster A and cluster B (and future clusters) settings from overlapping: the runner and
launchers read cluster-specific paths/accounts/caches from here instead of hardcoding one cluster's
values as shared defaults.

Selection order:
  1. ``$SP_CLUSTER`` (explicit: "cluster_a" | "cluster_b")
  2. auto-detect from a marker directory on disk (robust to hostnames like ``n05``)
  3. fallback: "cluster_a"

Every value is still overridable by its own env var at the point of use (e.g. ``$HF_HUB_CACHE``),
so this only sets *defaults* — it never overrides an explicit choice.

Shell usage (launchers):  eval "$(python -m ac2.clusters export)"
  -> exports SP_CLUSTER, SP_ACCOUNT, SP_PARTITION, SP_QOS, HF_HUB_CACHE, HF_HOME, SP_CACHE_ROOT,
     SP_JUDGE_MODEL, WANDB_MODE  (only non-empty ones), without clobbering already-set vars.
"""

import os
import sys

_USER = os.environ.get("USER", "sp")


def _site(var: str, suffix: str = "") -> str:
    """A site-specific value from the environment (see .env.example); empty when unset."""
    base = os.environ.get(var, "")
    return base + suffix if base else ""


CLUSTERS = {
    # cluster A (H100). Shared team HF cache in /scratch is READ-ONLY for us, so models are
    # read from HF_HUB_CACHE there while HF_HOME stays writable (datasets/parquet FileLocks). Heavy
    # COMPILE caches (Triton/inductor/vLLM) go node-local (/tmp) to avoid NFS ESTALE on the Triton
    # cache — those libs mkdir their own dirs, so a fresh node's empty /tmp is fine. The one cache
    # that must be SHARED is the tokenizer vocab (openai_harmony/tiktoken_rs for the gpt-oss judge):
    # tiktoken_rs does NOT mkdir its parent and crashes on a fresh node's empty /tmp — see
    # tiktoken_cache_dir below (shared home; the project scratch is quota-blocked for us).
    "cluster_a": {
        "account": _site("AC2_CLUSTER_A_ACCOUNT_ALT"),
        "partition": "preempt",
        "qos": "",
        "hf_hub_cache": _site("AC2_CLUSTER_A_ROOT", "/hf_home/hub"),
        "hf_home": _site("AC2_CLUSTER_A_ROOT", "/hf_home"),
        "cache_root": _site("AC2_CLUSTER_A_ROOT", "/.cache"),
        "tiktoken_cache_dir": os.path.expanduser("~/.cache/sp_cache/tiktoken_rs"),
        "judge_model": "openai/gpt-oss-120b",
        "wandb_mode": "online",
        "gpu_mem_gb": 80,            # H100 80GB (verified via nvidia-smi on a cluster A node)
        "ppo_max_token_len": 51200,  # 80GB: one 50k seq/backward (98304 OOMs the actor update)
        "rollout_max_num_seqs": 96,  # 80GB actor concurrency (H200's 192 OOMs here)
        "reward_max_num_seqs": 96,   # judge concurrency (128 OOMed the judge; lowered to 96)
        # DS4-Flash judge snapshot (same revision on both clusters; only the cache root
        # differs). runner.py reads this instead of hardcoding one cluster's path.
        "ds4_snapshot": _site("AC2_CLUSTER_A_ROOT", "/hf_home/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"),
        "marker": _site("AC2_CLUSTER_A_MARKER"),
    },
    # cluster B (H100 80GB). Its scratch is writable, so no read-only-cache split is needed
    # (HF_HUB_CACHE defaults to HF_HOME/hub); caches live on the fast parallel FS.
    "cluster_b": {
        "account": _site("AC2_CLUSTER_B_ACCOUNT"),
        "partition": _site("AC2_CLUSTER_B_PARTITION"),
        "qos": _site("AC2_CLUSTER_B_QOS"),
        "hf_hub_cache": "",  # empty -> use HF_HOME/hub (gpfs is writable)
        "hf_home": _site("AC2_CLUSTER_B_SCRATCH", "/.cache/huggingface"),
        "cache_root": _site("AC2_CLUSTER_B_ROOT", "/.cache"),
        "judge_model": "openai/gpt-oss-120b",
        "wandb_mode": "offline",
        "gpu_mem_gb": 80,            # H100 80GB
        "ppo_max_token_len": 51200,  # 80GB: one 50k seq/backward (98304 OOMs the actor update)
        "rollout_max_num_seqs": 96,  # scaled down from the H200 setting of 192 for 80GB
        "reward_max_num_seqs": 64,   # 128->96->64 (cluster B): a judge OOM during colocated init can stall a TP rank -> NCCL allreduce watchdog hang
        "ds4_snapshot": _site("AC2_CLUSTER_B_SCRATCH", "/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"),
        "marker": _site("AC2_CLUSTER_B_MARKER"),
    },
}

_ENV_KEYS = {  # cluster-config key -> env var it feeds
    "account": "SP_ACCOUNT",
    "partition": "SP_PARTITION",
    "qos": "SP_QOS",
    "hf_hub_cache": "HF_HUB_CACHE",
    "hf_home": "HF_HOME",
    "cache_root": "SP_CACHE_ROOT",
    "judge_model": "SP_JUDGE_MODEL",
    "wandb_mode": "WANDB_MODE",
}


def detect_cluster() -> str:
    """Return the cluster name: $SP_CLUSTER, else a marker-dir match, else 'cluster_a'."""
    name = os.environ.get("SP_CLUSTER")
    if name:
        return name
    for cname, cfg in CLUSTERS.items():
        marker = cfg.get("marker")
        if marker and os.path.isdir(marker):
            return cname
    return "cluster_a"


def cluster_config(name: str | None = None) -> dict:
    """Return the config dict for the selected (or auto-detected) cluster."""
    name = name or detect_cluster()
    if name not in CLUSTERS:
        raise ValueError(f"unknown SP_CLUSTER={name!r}; known: {sorted(CLUSTERS)}")
    return {**CLUSTERS[name], "name": name}


def _print_shell_exports() -> None:
    """Emit `export VAR=val` lines for a launcher to eval. Skips empty values and never clobbers a
    var already set in the environment (so a launcher/CLI override always wins)."""
    cfg = cluster_config()
    print(f'export SP_CLUSTER="{cfg["name"]}"')
    for key, env in _ENV_KEYS.items():
        val = cfg.get(key, "")
        if val and not os.environ.get(env):
            print(f'export {env}="{val}"')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        _print_shell_exports()
    else:
        import json
        print(json.dumps(cluster_config(), indent=2))
