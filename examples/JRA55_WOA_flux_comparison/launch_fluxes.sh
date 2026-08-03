#!/bin/bash
# launch_fluxes.sh — submit a JRA55 × WOA prescribed-ocean flux run to SLURM
# on the GCP a3mega cluster (companion of examples/OMIP_GCP/launch_inrepo.sh).
#
# The Julia script (prescribed_woa_fluxes.jl) is configured entirely through
# environment variables, which this launcher forwards into the job.
#
# Usage:
#   ./launch_fluxes.sh                          # corrected (COARE 3.6), 1958+5yr
#                                               # (same start/length as the seasonal-calibration forward runs)
#   FLUX_CONFIG=ncar ./launch_fluxes.sh         # Large & Yeager (OMIP-2 protocol)
#   STOP_YEARS=2 ./launch_fluxes.sh             # 2-year smoke test
#   SCRIPT_ONLY=true ./launch_fluxes.sh         # print the run command, no sbatch
#
# Knobs (defaults):
#   FLUX_CONFIG   corrected | ncar        (corrected)
#   START_YEAR    1958
#   STOP_YEARS    5
#   DT_HOURS      1.0
#   FORCING_DIR   /home/ext_xinkai_caltech_edu/JRA55_data
#   RESTORING_DIR /home/ext_xinkai_caltech_edu/ECCO_data
#   BACKEND_SIZE  240
#   THREADS       4
#   PARTITION     a3mega     TIME 6:00:00     MEM 100GB
#   INSTANTIATE   true
#   SCRIPT_ONLY   false

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIMA_CALIB_PROJECT="${CLIMA_CALIB_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export FLUX_CONFIG="${FLUX_CONFIG:-corrected}"
export START_YEAR="${START_YEAR:-1958}"
export STOP_YEARS="${STOP_YEARS:-5}"
export DT_HOURS="${DT_HOURS:-1.0}"
export FORCING_DIR="${FORCING_DIR:-/home/ext_xinkai_caltech_edu/JRA55_data}"
export RESTORING_DIR="${RESTORING_DIR:-/home/ext_xinkai_caltech_edu/ECCO_data}"
export BACKEND_SIZE="${BACKEND_SIZE:-240}"
export ARCH="${ARCH:-gpu}"
THREADS="${THREADS:-4}"

RUN_NAME="${RUN_NAME:-woafluxes_${FLUX_CONFIG}_${START_YEAR}_${STOP_YEARS}yr}"
export RUN_NAME
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_NAME}_run}"

LOG_DIR="${LOG_DIR:-${CLIMA_CALIB_PROJECT}/logs}"
mkdir -p "$LOG_DIR"

JULIA_CMD=(julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "$THREADS"
           "$SCRIPT_DIR/prescribed_woa_fluxes.jl")

if [[ "${SCRIPT_ONLY:-false}" == "true" ]]; then
    echo "Run with:"
    echo "  FLUX_CONFIG=$FLUX_CONFIG START_YEAR=$START_YEAR STOP_YEARS=$STOP_YEARS DT_HOURS=$DT_HOURS \\"
    echo "  FORCING_DIR=$FORCING_DIR RESTORING_DIR=$RESTORING_DIR ARCH=$ARCH \\"
    echo "  ${JULIA_CMD[*]}"
    exit 0
fi

sbatch \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task="$THREADS" \
    --partition="${PARTITION:-a3mega}" \
    --time="${TIME:-6:00:00}" \
    --mem="${MEM:-100GB}" \
    -J "$RUN_NAME" \
    -o "${LOG_DIR}/${RUN_NAME}.%j.out" -e "${LOG_DIR}/${RUN_NAME}.%j.out" \
    --export=ALL,REPEAT_YEAR="${REPEAT_YEAR:-false}",CLIMA_CALIB_PROJECT="$CLIMA_CALIB_PROJECT",THREADS="$THREADS",SCRIPT_DIR="$SCRIPT_DIR" \
    "$@" <<'EOF'
#!/bin/bash
#SBATCH -N 1
source "$HOME/env_nvhpc_25.7.sh"

export JULIA_CUDA_MEMORY_POOL=none
export DATADEPS_ALWAYS_ACCEPT=true
[ -f "$HOME/API_keys.sh" ] && source "$HOME/API_keys.sh"

if [[ "${INSTANTIATE:-true}" == "true" ]]; then
    julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --startup-file=no -e 'using Pkg; Pkg.instantiate()'
fi

julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "${THREADS}" \
    "$SCRIPT_DIR/prescribed_woa_fluxes.jl"
EOF

echo "Submitted $RUN_NAME (logs: ${LOG_DIR}/${RUN_NAME}.<jobid>.out)"
