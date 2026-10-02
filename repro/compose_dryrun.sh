#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source repro/env.sh
for method in grpo ac2 prefix; do
  python repro/local_runner.py --method "$method" --run-dir "runs/e5/compose-$method" --compose-only --smoke
done
