"""Detached approved sequence: R1 complete+accepted, then R2; never auto-retry failures."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil
from local_runner import judge_accounting
from r_launch import command
from run_bounded import remember_tree, finish_children
from surrogate_common import ROOT, URL, profile
from surrogate_long_acceptance import collect


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    out = args.directory.resolve()
    assert out.is_relative_to(ROOT / "runs/research")
    out.mkdir(parents=True, exist_ok=False)
    lease = (ROOT / "runs/research/surrogate-sequence.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    base_api = judge_accounting()
    assert len(base_api) == 547 and all(r["state"] == "complete" for r in base_api)
    assert json.loads((ROOT / "repro/receipts/s3-surrogate-r1.json").read_text())["accepted"] is True
    agreement = json.loads((ROOT / "repro/receipts/s2-surrogate-agreement.json").read_text())
    assert agreement["coverage_verified"] and agreement["summary"]["count"] == len(base_api)
    assert agreement["summary"]["profile"] == profile() and agreement["summary"]["api_calls_added"] == 0
    assert json.loads(Path(agreement["api_before"]).read_text()) == base_api
    service = json.loads((ROOT / "repro/receipts/s1-surrogate-active.json").read_text())
    assert service["accepted"] and service["summary"]["profile"] == profile()
    canonical = json.loads((ROOT / "repro/receipts/e3-canonical.json").read_text())["sha256"]
    source = [ROOT / "repro" / name for name in (
        "surrogate_chain.py", "surrogate_long_acceptance.py", "local_runner.py", "r_launch.py",
        "surrogate_common.py", "surrogate_reward.py", "strict_reward.py", "r_curves.py", "run_bounded.py")]
    tracked = subprocess.check_output(["git", "ls-files", "-z", "src", "experiments", "repro/r_protocol.json"], cwd=ROOT)
    source += [ROOT / os.fsdecode(name) for name in tracked.split(b"\0") if name]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source}
    write(out / "plan.json", {"authorization": "Richard approved 2026-10-02: R1 then R2 only; surrogate GPU0, training GPUs1-7",
                             "pid": os.getpid(), "create_time": psutil.Process().create_time(),
                             "profile": profile(), "source_sha256": hashes,
                             "failure_default": "Stop without retry or R2 launch; preserve every artifact."})
    for task in ("r1", "r2"):
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
        assert judge_accounting() == base_api
        for entry in (service["server"], service["gateway"]):
            process = psutil.Process(entry["pid"])
            assert process.create_time() == entry["create_time"] and process.is_running(), "pinned judge process changed"
        for name in ("train.parquet", "test.parquet", "val_map.json"):
            with (ROOT / "runs/e3/canonical" / name).open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == canonical[name], "canonical data changed"
        with urllib.request.urlopen(URL.removesuffix("/v1")+"/health", timeout=20) as response:
            assert json.load(response) == profile()
        run = ROOT / "runs/research" / task
        assert not (run / "metrics.jsonl").exists() and not (run / "checkpoints").exists(), "only a fresh long run is authorized here"
        if task == "r2":
            prior = json.loads((out / "r1/acceptance.json").read_text())
            assert prior["surrogate_acceptance"]["complete_steps"] == 200
            # Revalidate R1 from its raw outputs immediately before R2 admission.
            assert json.loads(json.dumps(collect(ROOT / "runs/research/r1", out / "r1"))) == prior
        launch = out / task
        launch.mkdir()
        cmd = command(task, "long", judge_backend="surrogate")
        write(launch / "api_before.json", base_api)
        started = time.time()
        with (launch / "stdout.log").open("x") as log, (launch / "gpu.csv").open("x") as gpu, (launch / "gpu.stderr").open("x") as err:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = (child.pid, psutil.Process(child.pid).create_time())
            owned = {}
            sampler = None
            try:
                remember_tree(identity, owned)
                sampler = subprocess.Popen(["nvidia-smi", "-i", "1,2,3,4,5,6,7",
                    "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "--loop=5"],
                    stdin=subprocess.DEVNULL, stdout=gpu, stderr=err, start_new_session=True)
                owned[(sampler.pid, psutil.Process(sampler.pid).create_time())] = os.pidfd_open(sampler.pid)
                write(launch / "launch.json", {"task": task, "command": cmd, "pid": child.pid,
                    "create_time": identity[1], "unix": started, "physical_gpus": list(range(1, 8)), "approved_long": True})
                while child.poll() is None:
                    remember_tree(identity, owned)
                    assert sampler.poll() is None, "GPU telemetry failed"
                    assert judge_accounting() == base_api, "external API accounting changed"
                    with (launch / "heartbeat.jsonl").open("a") as f:
                        f.write(json.dumps({"unix": time.time(), "pid": child.pid, "elapsed_s": time.time()-started})+"\n")
                    time.sleep(10)
            finally:
                error = sys.exc_info()[1]
                finish_children(child, identity, owned)
                if sampler is not None:
                    sampler.wait(timeout=2)
                result = {"returncode": child.returncode, "elapsed_s": time.time()-started,
                          "remaining_owned_processes": 0, "monitor_error": None if error is None else repr(error)}
                write(launch / "result.json", result)
            assert child.returncode == 0, "long run failed; no retry or next task"
        acceptance = collect(run, launch)
        write(launch / "acceptance.json", acceptance)
        print(f"ACCEPTED {task} 200 steps", flush=True)
    write(out / "complete.json", {"tasks": ["r1", "r2"], "api_calls_added": 0})


if __name__ == "__main__":
    main()
