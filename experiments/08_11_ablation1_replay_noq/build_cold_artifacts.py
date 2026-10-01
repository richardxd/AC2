"""build_cold_artifacts.py -- write the EMPTY replay-buffer seed a from-scratch Prefix GRPO run
starts from. There is no critic, so unlike AC2's builder this writes no critic-buffer seed and
no reference bank.

`sp_replay._load_seed` requires a manifest and at least one shard file even when there are no
records (an empty shard is legal, zero shards is not). This script writes exactly that, with
the same manifest/shard/sha layout as a non-empty seed, so the sha pin, `verify_manifest_shards`
and the state-file drift checks behave identically.

With an empty buffer step 1 has nothing to cut; with `sp_replay_cold_bootstrap=1` its replay
slots are filled with fresh `cold_scratch` problems (full group size, trained), and the inflow
lane fills the buffer from step 1's own trajectories (ungated admission). From step 2 the
global draw takes `n_replay` picks with wraparound over however few entries exist.

Run once before the first launch:

  python experiments/08_11_ablation1_replay_noq/build_cold_artifacts.py \
      --out experiments/08_11_ablation1_replay_noq

Idempotent (no RNG and no input data, so the manifest sha does not move and a resume cannot trip
a drift check); refuses to overwrite a non-empty seed unless --force is given.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "08_11_ablation1_replay_noq",
    ),
)
from q_build_common import sha256_file, write_shards  # noqa: E402


def _existing_record_count(out: str, subdir: str) -> int:
    n = 0
    for path in glob.glob(os.path.join(out, subdir, "shard_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            n += sum(1 for line in f if line.strip())
    return n


def _clear_shards(base: str, subdir: str) -> int:
    """Delete every existing shard_*.jsonl before writing.

    `write_shards` writes shard_0000..shard_N for the records it is
    given and NEVER deletes anything, so rebuilding a previously multi-shard artifact as an
    empty one would leave shard_0001+ in place: their records would still be loaded (the
    loaders glob the directory), and they are not in the new manifest, so
    verify_manifest_shards fails -- or worse, an older loader picks them up. Clearing first
    makes the rebuild actually a rebuild."""
    removed = 0
    for path in sorted(glob.glob(os.path.join(base, subdir, "shard_*.jsonl"))):
        os.remove(path)
        removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="experiment dir (holds replay_seed_cold/, q_seed/, reference_bank/)")
    ap.add_argument("--replay-seed-name", default="replay_seed_cold",
                    help="subdir for the empty replay seed; must match SP_REPLAY_SEED_DIR")
    ap.add_argument("--fifo-cap", type=int, default=1920,
                    help="recorded in the manifest for provenance only; the FIFO capacity the "
                         "run enforces is SP_Q_FIFO_CAP (20-step horizon x 96 groups = 1920)")
    ap.add_argument("--min-valid", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if an existing seed/bank has records")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    replay_out = os.path.join(out, args.replay_seed_name)

    for base, subdir in ((replay_out, "replay_buffer"),):
        have = _existing_record_count(base, subdir)
        if have and not args.force:
            raise SystemExit(
                f"refusing to overwrite {base}/{subdir}: it already holds {have} records "
                "(pass --force only if you really mean to discard them)"
            )

    # Clear any pre-existing shards first (see _clear_shards): write_shards only overwrites
    # shard_0000, so a stale shard_0001+ would survive a "rebuild" and be loaded anyway.
    for _base, _sub in ((replay_out, "replay_buffer"),):
        _rm = _clear_shards(_base, _sub)
        if _rm:
            print(f"[cold] cleared {_rm} pre-existing shard(s) from {_base}/{_sub}", flush=True)

    # ---- empty REPLAY buffer seed --------------------------------------------------------
    # Same layout sp_replay._load_seed expects: replay_buffer_manifest.json + at least one
    # replay_buffer/shard_*.jsonl. Zero entries means the run has no trajectory to cut until
    # its own inflow lane admits one, which is what sp_replay_cold_bootstrap=1 permits.
    os.makedirs(replay_out, exist_ok=True)
    rshards = write_shards([], replay_out, "replay_buffer")
    rmanifest = {
        "builder": "build_cold_artifacts.py",
        "cold_start": True,
        "reason": "truly empty buffer: the run bootstraps its own trajectories from step 1",
        "entries": 0,
        "problems": 0,
        "capacity": None,
        "shards": rshards,
    }
    rpath = os.path.join(replay_out, "replay_buffer_manifest.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(rmanifest, f, indent=2)
    print(f"[cold] empty replay seed -> {rpath}\n  sha256 = {sha256_file(rpath)}", flush=True)

    # ---- empty Q FIFO seed ----
    # NO critic-buffer seed or reference bank: this configuration has no critic, so there is
    # nothing to seed and no loader that requires a manifest. AC2's builder writes both here.


if __name__ == "__main__":
    main()
