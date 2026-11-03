#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=prefetch
#SBATCH --output=logs/%x.%j.out

# Usage: sbatch examples/Arctic_sea_ice/run_prefetch.slurm [prefetch_glorys.jl options]
# Run from the repository root, once per job index: --days 365 --njobs 8 --job 0, …, --job 7.

source ~/env_nvhpc_26.5.sh
source ~/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL

EXAMPLE_DIR="$SLURM_SUBMIT_DIR/examples/Arctic_sea_ice"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), started $(date)"
echo "Options: $*"

julia +1.13.3 --project="$EXAMPLE_DIR" -e 'using Pkg; Pkg.instantiate()'
stdbuf -oL -eL julia +1.13.3 -t 16 --project="$EXAMPLE_DIR" "$EXAMPLE_DIR/prefetch_glorys.jl" "$@"

echo "Finished $(date)"
