#!/bin/bash
# submit_train_chunks.sh — submit N chained gpu-dev training chunks.
#
# Each chunk is 3h50m on gpu-dev (2 nodes x 8 GPUs); the run name is FIXED so
# chunk k+1 auto-resumes from chunk k's last checkpoint. Chunks are chained
# with Slurm --dependency=afterany, so the next chunk starts as soon as the
# previous one ends (including the expected wall-clock kill at 3h50m).
#
# Usage:
#   submit_train_chunks.sh <model> <n_chunks> [--begin HH:MM[:SS]]
#
#   model     : small | base | large
#   n_chunks  : how many 3h50m chunks to submit in a row (>=1)
#   --begin   : (optional) delay the FIRST chunk to HH:MM (e.g. 03:30 to skip
#               the overnight weather-model inference); the rest chain after it
#
# Examples:
#   submit_train_chunks.sh base 1                  # one chunk (this round)
#   submit_train_chunks.sh base 3                  # 3 chunks back-to-back
#   submit_train_chunks.sh base 3 --begin 03:30    # start after 03:30

set -euo pipefail
mkdir -p slurm-logs

if [ "$#" -lt 2 ]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi

MODEL="$1"
N="$2"
BEGIN=""
shift 2
while [ "$#" -gt 0 ]; do
  case "$1" in
    --begin) BEGIN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$MODEL" in small|base|large) ;; *) echo "ERROR: model must be small|base|large" >&2; exit 2 ;; esac
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
  echo "ERROR: n_chunks must be a positive integer (got '$N')" >&2; exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/train_paper_dev.sbatch"

prev=""
first=""
for ((i=1; i<=N; i++)); do
  args=(--parsable --export=ALL,MODEL="$MODEL")
  if [ -z "$prev" ]; then
    [ -n "$BEGIN" ] && args+=(--begin="$BEGIN")
  else
    args+=(--dependency=afterany:"$prev")
  fi
  jid=$(sbatch "${args[@]}" "$SBATCH_SCRIPT")
  [ -z "$first" ] && first="$jid"
  echo "chunk $i/$N: job $jid"
  prev="$jid"
done

echo "----"
echo "MODEL=$MODEL  chunks=$N  first_job=$first  last_job=$prev"
