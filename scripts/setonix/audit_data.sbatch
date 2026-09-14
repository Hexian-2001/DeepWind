#!/bin/bash --login
#SBATCH --job-name=deepwind-data-audit
#SBATCH --partition=copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --account=pawsey0115

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3

: "${DEEPWIND_PROJECT_ROOT:?Set DEEPWIND_PROJECT_ROOT}"
: "${DEEPWIND_DATA_ROOT:?Set DEEPWIND_DATA_ROOT}"
: "${DEEPWIND_REPORTS_ROOT:?Set DEEPWIND_REPORTS_ROOT}"
: "${DEEPWIND_VENV:?Set DEEPWIND_VENV}"

mkdir -p "${DEEPWIND_REPORTS_ROOT}"
stamp="$(date -u +%Y-%m-%dT%H%M%SZ)"
manifest="${DEEPWIND_REPORTS_ROOT}/corpus-files-${stamp}.tsv"
report="${DEEPWIND_REPORTS_ROOT}/dataset-full-audit-${stamp}.json"

for split in train eval; do
  find "${DEEPWIND_DATA_ROOT}/${split}" -type f -name '*.npy' \
    -printf "${split}/%f\t%s\n"
done | LC_ALL=C sort > "${manifest}"

sha256sum \
  "${DEEPWIND_DATA_ROOT}/train_metadata.csv" \
  "${DEEPWIND_DATA_ROOT}/eval_metadata.csv" \
  "${manifest}" > "${DEEPWIND_REPORTS_ROOT}/manifest-checksums-${stamp}.txt"

singularity exec \
  --bind "${DEEPWIND_PROJECT_ROOT},${DEEPWIND_DATA_ROOT},${DEEPWIND_REPORTS_ROOT},${DEEPWIND_VENV}" \
  "${SINGULARITY_CONTAINER}" \
  "${DEEPWIND_VENV}/bin/python" \
  "${DEEPWIND_PROJECT_ROOT}/tools/audit_dataset.py" \
  --data-root "${DEEPWIND_DATA_ROOT}" \
  --output "${report}" \
  --sample-size 1000000000

printf 'manifest=%s\nreport=%s\n' "${manifest}" "${report}"
