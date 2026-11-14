#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name=gm_tapering
#SBATCH --output=logs/%x_%j.out

# sbatch --export=ALL,MAX_SLOPE=0.01 examples/GM_tapering_sensitivity/run_gm_tapering.slurm  (from the repository root)

MAX_SLOPE=${MAX_SLOPE:-0.01}
YEARS=${YEARS:-20}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-$HOME/gm_tapering_sensitivity}
PROJECT=${PROJECT:-.}

source $HOME/env_nvhpc_26.5.sh
source $HOME/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL
export JULIA_NUM_THREADS=16

julia +1.12 --project=$PROJECT examples/GM_tapering_sensitivity/run_gm_tapering.jl \
    --max-slope $MAX_SLOPE --years $YEARS --output-directory $OUTPUT_DIRECTORY ${PICKUP:+--pickup $PICKUP}
