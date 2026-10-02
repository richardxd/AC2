"""Approved S3 bounded local-surrogate smoke, preserving prior DeepSeek artifacts."""
import argparse
import subprocess
import sys
from surrogate_common import ROOT
from r_launch import command


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["cold", "resume"])
    args = p.parse_args()
    cmd = command("r1", "smoke-"+args.phase, judge_backend="surrogate")
    subprocess.run([sys.executable, str(ROOT / "repro/run_bounded.py"), "--seconds", "1740",
                    "--receipt-dir", str(ROOT / f"runs/surrogate/s3-{args.phase}01"), "--", *cmd], check=True)
