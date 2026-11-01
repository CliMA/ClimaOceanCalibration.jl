#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --job-name=glorys_series
#SBATCH --output=logs/%x.%j.out

# Usage: sbatch examples/Arctic_sea_ice/analysis/run_glorys_series.slurm <ice or surface> <run dir> <start date> <ndays>
# Run from the repository root. Files not yet in the GLORYS cache download about one a minute; fetch a year in
# parallel first (prefetch_glorys.jl) or allow a long time limit.

source ~/env_nvhpc_26.5.sh
source ~/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL

SERIES=$1; shift
EXAMPLE_DIR="$SLURM_SUBMIT_DIR/examples/Arctic_sea_ice"
julia +1.13.3 -t 16 --project="$EXAMPLE_DIR" "$EXAMPLE_DIR/analysis/glorys_${SERIES}_series.jl" "$@"
echo "Finished $(date)"
