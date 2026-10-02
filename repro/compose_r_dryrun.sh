#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source repro/env.sh
for task in r1 r2 r3 r4-2k r4-correct-only r4-no-audit; do
  python repro/r_launch.py "$task" compose
done
