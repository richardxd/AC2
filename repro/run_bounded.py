"""Bound one local engineering run and capture physical-GPU telemetry.

Only signal captured descendants of this launch via identity-checked pidfds.
A timed-out run is a failed smoke, not a success.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def remember_tree(root_identity, owned):
    """Keep pidfds for this launch's descendants; pid reuse cannot redirect signals."""
    processes = []
    for pid, born in {root_identity, *owned.keys()}:
        try:
            parent = psutil.Process(pid)
            if parent.create_time() != born:
                continue
            processes.append(parent)
            processes.extend(parent.children(recursive=True))
        except psutil.NoSuchProcess:
            continue
    for process in processes:
        fd = None
        try:
            identity = (process.pid, process.create_time())
            if identity not in owned and process.is_running():
                fd = os.pidfd_open(process.pid)
                if psutil.Process(process.pid).create_time() != identity[1]:
                    os.close(fd)
                    raise RuntimeError("child PID changed during capture")
                owned[identity] = fd
        except (psutil.NoSuchProcess, ProcessLookupError):
            if fd is not None:
                os.close(fd)


def active(owned):
    result = []
    for (pid, born), fd in owned.items():
        try:
            p = psutil.Process(pid)
            if p.create_time() == born and p.status() != psutil.STATUS_ZOMBIE:
                result.append(fd)
        except psutil.NoSuchProcess:
            pass
    return result


def finish_children(child, root_identity, owned):
    remember_tree(root_identity, owned)
    # Check descendants even when the leader exited normally: Ray may start
    # workers in separate sessions. pidfds identify only this captured tree.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 10
        while active(owned) and time.monotonic() < deadline:
            remember_tree(root_identity, owned)
            for fd in active(owned):
                try:
                    signal.pidfd_send_signal(fd, sig)
                except ProcessLookupError:
                    pass
            time.sleep(.2)
    child.wait(timeout=2)
    remaining = len(active(owned))
    for fd in owned.values():
        os.close(fd)
    assert remaining == 0, f"{remaining} launched processes still alive"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=1500)
    p.add_argument("--receipt-dir", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    assert 1 <= args.seconds <= 1740
    root = Path(__file__).resolve().parents[1]
    out = args.receipt_dir.resolve()
    assert out.is_relative_to(root / "runs")
    out.mkdir(parents=True, exist_ok=False)
    cmd = args.command
    if cmd[0] == "--":
        cmd = cmd[1:]
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert gpus and all(x in {"1", "2", "3", "4", "5", "6", "7"} for x in gpus)
    start = time.monotonic()
    with (out / "stdout.log").open("x") as log, (out / "gpu.csv").open("x") as gpu_log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # The child has not been reaped, so its PID cannot have been reused.
        root_identity = (child.pid, psutil.Process(child.pid).create_time())
        (out / "launch.json").write_text(json.dumps({"pid": child.pid, "process_group": child.pid,
            "command": cmd, "gpus": gpus, "seconds": args.seconds, "unix": time.time()}, indent=2))
        timed_out = False
        owned = {}
        try:
            while child.poll() is None:
                remember_tree(root_identity, owned)
                sample = subprocess.run(["nvidia-smi", "-i", ",".join(gpus),
                    "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
                gpu_log.write(sample.stdout)
                gpu_log.flush()
                if time.monotonic() - start >= args.seconds:
                    timed_out = True
                    break
                time.sleep(1)
        finally:
            finish_children(child, root_identity, owned)
        result = {"returncode": child.returncode, "timed_out": timed_out, "elapsed_s": time.monotonic() - start}
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        sys.exit(124 if timed_out else child.returncode)


if __name__ == "__main__":
    main()
