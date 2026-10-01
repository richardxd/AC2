#!/usr/bin/env bash
# sbatch_env.sh FILE [SBATCH_OPTIONS...] -- submit an experiment's SLURM script after filling in
# the ${AC2_*} site settings (account, partition, QOS, log paths, ...) in its #SBATCH header.
# SLURM does not expand environment variables in #SBATCH lines, so this substitutes them from the
# repository's .env (see .env.example) and pipes the result to sbatch, which exports the same
# environment to the job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
f="${1:?usage: sbatch_env.sh FILE [SBATCH_OPTIONS...]}"; shift
perl -pe 's/\$\{(AC2_[A-Z0-9_]+)\}/defined $ENV{$1} ? $ENV{$1} : die "sbatch_env.sh: $1 is not set (see .env.example)\n"/ge' "$f" \
  | sbatch "$@"
