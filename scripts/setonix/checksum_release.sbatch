#!/bin/bash --login
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --job-name=deepwind-checksum
#SBATCH --partition=copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_PROJECT

set -euo pipefail

: "${DEEPWIND_RELEASE_DIR:?Set DEEPWIND_RELEASE_DIR}"
cd "$DEEPWIND_RELEASE_DIR"

find -L . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS

sha256sum --check SHA256SUMS
