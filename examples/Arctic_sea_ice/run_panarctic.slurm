#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=panarctic
#SBATCH --output=logs/%x.%j.out

# Usage: sbatch examples/Arctic_sea_ice/run_panarctic.slurm [config.toml]
# Run from the repository root. Without an argument the example's config.toml is used.

source ~/env_nvhpc_26.5.sh
source ~/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL

EXAMPLE_DIR="$SLURM_SUBMIT_DIR/examples/Arctic_sea_ice"
CONFIG="${1:-$EXAMPLE_DIR/config.toml}"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), started $(date)"
echo "Config: $CONFIG"
nvidia-smi -L

julia +1.13.3 --project="$EXAMPLE_DIR" -e 'using Pkg; Pkg.instantiate()'
stdbuf -oL -eL julia +1.13.3 --project="$EXAMPLE_DIR" "$EXAMPLE_DIR/panarctic.jl" "$CONFIG"

echo "Finished $(date)"
