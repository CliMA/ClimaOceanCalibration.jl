#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=104
#SBATCH --mem=0
#SBATCH --job-name=catke_diffusivity
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

# sbatch --export=ALL,DIFFUSIVITY_SCALES="0.5 1 2" examples/CATKE_diffusivity_sensitivity/run_catke_diffusivity.slurm  (from the repository root)
# One run per GPU of the node. A requeued job continues every run from the latest checkpoint in its output directory.

DIFFUSIVITY_SCALES=${DIFFUSIVITY_SCALES:-0.5 1 2}
YEARS=${YEARS:-20}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-$HOME/catke_diffusivity_sensitivity}
PROJECT=${PROJECT:-.}

source $HOME/env_nvhpc_26.5.sh
source $HOME/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL
export JULIA_NUM_THREADS=16

gpu=0
for scale in $DIFFUSIVITY_SCALES; do
    CUDA_VISIBLE_DEVICES=$gpu julia +1.12 --project=$PROJECT examples/CATKE_diffusivity_sensitivity/run_catke_diffusivity.jl \
        --diffusivity-scale $scale --years $YEARS --output-directory $OUTPUT_DIRECTORY \
        >> logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_scale_$scale.out 2>&1 &
    gpu=$((gpu + 1))
done
wait
