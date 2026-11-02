#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=panarctic
#SBATCH --output=logs/%x.%j.out

# Usage: sbatch examples/Arctic_sea_ice/run_panarctic.slurm [panarctic.jl options]
# Run from the repository root, e.g. with --stop_hours 3 --output_dir output/quick for a quick check.

source ~/env_nvhpc_26.5.sh
source ~/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL

EXAMPLE_DIR="$SLURM_SUBMIT_DIR/examples/Arctic_sea_ice"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), started $(date)"
echo "Options: $*"
nvidia-smi -L

julia +1.13.3 --project="$EXAMPLE_DIR" -e 'using Pkg; Pkg.instantiate()'
stdbuf -oL -eL julia +1.13.3 -t 16 --project="$EXAMPLE_DIR" "$EXAMPLE_DIR/panarctic.jl" "$@"

echo "Finished $(date)"
