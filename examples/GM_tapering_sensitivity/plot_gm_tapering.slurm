#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --job-name=gm_tapering_figures
#SBATCH --output=logs/%x_%j.out

# sbatch --export=ALL,FIRST_YEAR=15,LAST_YEAR=20 examples/GM_tapering_sensitivity/plot_gm_tapering.slurm  (from the repository root)

FIRST_YEAR=${FIRST_YEAR:-15}
LAST_YEAR=${LAST_YEAR:-20}
RUNS_DIRECTORY=${RUNS_DIRECTORY:-$HOME/gm_tapering_sensitivity}
FIGURE_DIRECTORY=${FIGURE_DIRECTORY:-$RUNS_DIRECTORY/figures}
PROJECT=${PROJECT:-.}
MAX_SLOPES=${MAX_SLOPES:-0.001 0.01 0.1}

source $HOME/env_nvhpc_26.5.sh
export JULIA_NUM_THREADS=32

julia +1.12 --project=$PROJECT examples/GM_tapering_sensitivity/plot_gm_tapering.jl \
    --runs-directory $RUNS_DIRECTORY --figure-directory $FIGURE_DIRECTORY \
    --first-year $FIRST_YEAR --last-year $LAST_YEAR --max-slopes $MAX_SLOPES
