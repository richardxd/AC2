#!/usr/bin/env python3
"""Sync training checkpoints & plot caches BETWEEN SERVERS via PRIVATE HuggingFace repos.

cluster A and cluster B can't scp to each other, but both have outbound internet and
``huggingface_hub``. This routes big artifacts through private HF repos so a checkpoint
produced on one server can be pulled into the *same experiment folder* on another, and a
run's plotting cache can be moved so you can render the dashboard on a second server.

Layout on HF (all repos PRIVATE, owned by your HF account):
  * checkpoint  ->  one repo per checkpoint:  <user>/sp-ckpt-<exp>-step<N>   (repo_type=model)
                    the folder run_data/checkpoints/global_step_<N>/ is mirrored to the repo root.
  * plot cache  ->  one shared repo:          <user>/sp-plotcache            (repo_type=dataset)
                    each experiment lives under  <exp>/analysis/*_fig_data.json  +  <exp>/dashboard_annotations.json

One-time per server:  `setup-token`  writes your HF token to ~/.cache/huggingface/token.

Usage (run from anywhere; the repo root is auto-detected, or set SELF_PLAY_ROOT):
  python -m ac2.utils.hf_ckpt_sync setup-token [--token TOK]   # default reads $HF_TOKEN
  python -m ac2.utils.hf_ckpt_sync whoami                      # auth check
  python -m ac2.utils.hf_ckpt_sync push-ckpt <exp> <step>      # upload a checkpoint (private)
  python -m ac2.utils.hf_ckpt_sync pull-ckpt <exp> <step>      # download it into experiments/<exp>/...
  python -m ac2.utils.hf_ckpt_sync push-plot <exp>            # upload the plotting cache
  python -m ac2.utils.hf_ckpt_sync pull-plot <exp>            # download the plotting cache
  python -m ac2.utils.hf_ckpt_sync list-ckpt <exp>            # list checkpoints pushed for an exp

Override the HF account / repo names with SP_HF_USER and SP_HF_PLOT_REPO if needed.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Locate the repo root: $SELF_PLAY_ROOT, else walk up from this file until a
    dir contains an ``experiments/`` subdir."""
    env = os.environ.get("SELF_PLAY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    p = Path(__file__).resolve()
    for cand in [p, *p.parents]:
        if (cand / "experiments").is_dir():
            return cand
    # src/ac2/utils/hf_ckpt_sync.py -> parents[3] is the repo root
    return p.parents[3]


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _user(api) -> str:
    u = os.environ.get("SP_HF_USER")
    if u:
        return u
    return api.whoami()["name"]


def _sanitize(s: str) -> str:
    # HF repo names allow [A-Za-z0-9._-]; exp names use [A-Za-z0-9_] already.
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def _ckpt_repo_id(user: str, exp: str, step) -> str:
    return f"{user}/sp-ckpt-{_sanitize(exp)}-step{step}"


def _plot_repo_id(user: str) -> str:
    return os.environ.get("SP_HF_PLOT_REPO") or f"{user}/sp-plotcache"


def _ckpt_dir(root: Path, exp: str, step) -> Path:
    return root / "experiments" / exp / "run_data" / "checkpoints" / f"global_step_{step}"


def _upload_folder_robust(api, *, repo_id, folder_path, repo_type, ignore_patterns=None):
    """Prefer upload_large_folder (resumable, parallel, xet/LFS — right for ~47GB checkpoints);
    fall back to upload_folder on older huggingface_hub."""
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo_id, folder_path=str(folder_path), repo_type=repo_type,
                                private=True, ignore_patterns=ignore_patterns)
    else:
        api.upload_folder(repo_id=repo_id, folder_path=str(folder_path), repo_type=repo_type,
                          commit_message="hf_ckpt_sync upload", ignore_patterns=ignore_patterns)


# ---------------------------------------------------------------------------

def cmd_setup_token(args):
    from huggingface_hub import login
    tok = args.token or os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("no token: pass --token or export HF_TOKEN")
    login(token=tok, add_to_git_credential=False)  # writes ~/.cache/huggingface/token
    name = _api().whoami(token=tok)["name"]
    print(f"[hf-sync] stored HF token for user '{name}' -> ~/.cache/huggingface/token")


def cmd_whoami(args):
    who = _api().whoami()
    print(f"[hf-sync] HF user: {who['name']}  (repo owner for pushes)")


def cmd_push_ckpt(args):
    root = _repo_root()
    # --src: push from an arbitrary directory (normally run_data/protected_checkpoints/
    # global_step_<N>_frozen). The LIVE checkpoint dir is retention-managed — with
    # max_actor_ckpt_to_keep=5 at ~1 step/hour, step N is deleted ~5h after it lands, which
    # is well inside a 92GB upload window. Uploading the live dir races retention and yields
    # a half-written repo; always push a frozen copy for anything that takes hours.
    src = Path(args.src).expanduser().resolve() if getattr(args, "src", None) \
        else _ckpt_dir(root, args.exp, args.step)
    if not src.is_dir():
        sys.exit(f"checkpoint not found: {src}")
    ignore = None
    if getattr(args, "skip_optimizer", False):
        # Optimizer moments are ~80% of the bytes (2x fp32 copies vs 1x bf16 model). Skipping them
        # means the receiver resumes with FRESH Adam moments (brief update-scale transient —
        # acceptable for a continuation at constant small lr). The tiny param_groups (lr/betas/
        # weight_decay/eps) are preserved as JSON so reshard --fresh-optim rebuilds the optimizer
        # with the exact source hyperparameters.
        import torch
        opt0 = sorted(src.glob("actor/optim_world_size_*_rank_0.pt"))
        if opt0:
            pg = torch.load(opt0[0], map_location="cpu", weights_only=False)["param_groups"]
            pg = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()} for g in pg]
            (src / "actor" / "optim_param_groups.json").write_text(json.dumps(pg, default=str))
            print(f"[hf-sync] --skip-optimizer: wrote actor/optim_param_groups.json from {opt0[0].name}")
        ignore = ["**/optim_world_size_*"]
    api = _api()
    repo_id = _ckpt_repo_id(_user(api), args.exp, args.step)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    sz = sum(
        f.stat().st_size for f in src.rglob("*")
        if f.is_file() and not (ignore and f.name.startswith("optim_world_size_"))
    ) / 1e9
    print(f"[hf-sync] pushing {src}  (~{sz:.1f} GB{', optimizer EXCLUDED' if ignore else ''})  ->  {repo_id}  (PRIVATE)")
    _upload_folder_robust(api, repo_id=repo_id, folder_path=src, repo_type="model", ignore_patterns=ignore)
    print(f"[hf-sync] DONE -> https://huggingface.co/{repo_id}")


def cmd_pull_ckpt(args):
    from huggingface_hub import snapshot_download
    root = _repo_root()
    dst = _ckpt_dir(root, args.exp, args.step)
    api = _api()
    repo_id = _ckpt_repo_id(_user(api), args.exp, args.step)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[hf-sync] pulling {repo_id}  ->  {dst}")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(dst))
    print(f"[hf-sync] DONE: checkpoint restored at {dst}")
    latest = dst.parent / "latest_checkpointed_iteration.txt"
    print(f"[hf-sync] note: this does NOT update {latest.name}. If you want the trainer to resume "
          f"from step {args.step}, set it: echo {args.step} > {latest}")


def cmd_push_plot(args):
    root = _repo_root()
    exp = args.exp
    adir = root / "experiments" / exp / "analysis"
    if not adir.is_dir():
        sys.exit(f"no analysis dir (nothing to plot from): {adir}")
    api = _api()
    repo_id = _plot_repo_id(_user(api))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"[hf-sync] pushing plot cache {adir}  ->  {repo_id}:{exp}/analysis  (PRIVATE)")
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(adir),
                      path_in_repo=f"{exp}/analysis",
                      allow_patterns=["*fig_data.json", "*.json", ".last_full_step"],
                      commit_message=f"plot cache {exp}")
    ann = root / "experiments" / exp / "dashboard_annotations.json"
    if ann.exists():
        api.upload_file(path_or_fileobj=str(ann), path_in_repo=f"{exp}/dashboard_annotations.json",
                        repo_id=repo_id, repo_type="dataset")
        print(f"[hf-sync]   + dashboard_annotations.json")
    print(f"[hf-sync] DONE -> https://huggingface.co/datasets/{repo_id}/tree/main/{exp}")


def cmd_pull_plot(args):
    from huggingface_hub import snapshot_download
    root = _repo_root()
    exp = args.exp
    api = _api()
    repo_id = _plot_repo_id(_user(api))
    # repo paths are <exp>/analysis/... and <exp>/dashboard_annotations.json; downloading into
    # experiments/ places them at experiments/<exp>/analysis/ and experiments/<exp>/ exactly.
    dest = root / "experiments"
    print(f"[hf-sync] pulling plot cache {repo_id}:{exp}/*  ->  {dest / exp}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[f"{exp}/*"],
                      local_dir=str(dest))
    fig = dest / exp / "analysis"
    print(f"[hf-sync] DONE. Render with:\n"
          f"    python -m ac2.viz.render_dashboard "
          f"{fig}/{exp}_fig_data.json {fig}")


def _data_repo_id(user: str) -> str:
    return f"{user}/sp-datasets"


def cmd_push_data(args):
    """Mirror one experiment-relative FILE to a shared private dataset repo.

    push-plot is scoped to fig_data by design, and push-ckpt expects a checkpoint directory
    layout, so neither moves an arbitrary artifact -- a training dataset, a token-id dump -- and
    the alternative is hand-rolled huggingface_hub calls on both ends with no shared convention
    for where things land. Keyed by <exp>/<relpath> so the file arrives at the SAME repo-relative
    path on the far side, which is what makes pull-data a no-argument-guessing operation.
    """
    root = _repo_root()
    src = Path(args.path)
    if not src.is_absolute():
        src = root / "experiments" / args.exp / args.path
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    api = _api()
    repo_id = _data_repo_id(_user(api))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    rel = args.name or src.name
    sz = src.stat().st_size / 1e6
    print(f"[hf-sync] pushing {src} ({sz:.0f} MB) -> {repo_id}:{args.exp}/{rel}  (PRIVATE)")
    api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{args.exp}/{rel}",
                    repo_id=repo_id, repo_type="dataset",
                    commit_message=f"{args.exp}: {rel}")
    print(f"[hf-sync] DONE -> https://huggingface.co/datasets/{repo_id}/tree/main/{args.exp}")


def cmd_pull_data(args):
    from huggingface_hub import hf_hub_download
    root = _repo_root()
    api = _api()
    repo_id = _data_repo_id(_user(api))
    # The repo path is ALREADY "<exp>/<name>", so the local_dir must be experiments/ and not
    # experiments/<exp> -- the latter lands the file at experiments/<exp>/<exp>/<name>. Same
    # convention pull-plot uses, for the same reason.
    dest_dir = Path(args.out) if args.out else (root / "experiments")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hf-sync] pulling {repo_id}:{args.exp}/{args.name} -> {dest_dir / args.exp}")
    # local_dir puts a real file there rather than a symlink into the hub cache, which matters
    # when the far side's HF_HOME is a shared group dir that may be pruned independently.
    p = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                        filename=f"{args.exp}/{args.name}", local_dir=str(dest_dir))
    print(f"[hf-sync] DONE -> {p}")


def cmd_list_ckpt(args):
    api = _api()
    user = _user(api)
    pat = re.compile(rf"^{re.escape(user)}/sp-ckpt-{re.escape(_sanitize(args.exp))}-step(\d+)$")
    found = []
    for m in api.list_models(author=user, search="sp-ckpt"):
        g = pat.match(m.id)
        if g:
            found.append(int(g.group(1)))
    if not found:
        print(f"[hf-sync] no checkpoints on HF for exp '{args.exp}'")
    else:
        print(f"[hf-sync] checkpoints on HF for '{args.exp}': steps {sorted(found)}")


def main():
    p = argparse.ArgumentParser(prog="hf_ckpt_sync", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup-token"); s.add_argument("--token", default=None); s.set_defaults(fn=cmd_setup_token)
    s = sub.add_parser("whoami"); s.set_defaults(fn=cmd_whoami)
    s = sub.add_parser("push-ckpt"); s.add_argument("exp"); s.add_argument("step")
    s.add_argument("--src", default=None,
                   help="push from this directory instead of run_data/checkpoints/global_step_<step> "
                        "(use the frozen copy: retention deletes the live dir mid-upload)")
    s.add_argument("--skip-optimizer", action="store_true",
                   help="exclude actor/optim_world_size_* shards (~80%% of bytes); receiver resumes "
                        "with fresh Adam moments (reshard --fresh-optim); param_groups preserved as JSON")
    s.set_defaults(fn=cmd_push_ckpt)
    s = sub.add_parser("pull-ckpt"); s.add_argument("exp"); s.add_argument("step"); s.set_defaults(fn=cmd_pull_ckpt)
    s = sub.add_parser("push-plot"); s.add_argument("exp"); s.set_defaults(fn=cmd_push_plot)
    s = sub.add_parser("pull-plot"); s.add_argument("exp"); s.set_defaults(fn=cmd_pull_plot)
    s = sub.add_parser("push-data"); s.add_argument("exp"); s.add_argument("path")
    s.add_argument("--name", default=None, help="name in the repo (default: the file's basename)")
    s.set_defaults(fn=cmd_push_data)
    s = sub.add_parser("pull-data"); s.add_argument("exp"); s.add_argument("name")
    s.add_argument("--out", default=None, help="destination dir (default experiments/<exp>)")
    s.set_defaults(fn=cmd_pull_data)
    s = sub.add_parser("list-ckpt"); s.add_argument("exp"); s.set_defaults(fn=cmd_list_ckpt)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
