#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --job-name=catke_diffusivity_figures
#SBATCH --output=logs/%x_%j.out

# sbatch --export=ALL,FIRST_YEAR=15,LAST_YEAR=20 examples/CATKE_diffusivity_sensitivity/plot_catke_diffusivity.slurm  (from the repository root)

FIRST_YEAR=${FIRST_YEAR:-15}
LAST_YEAR=${LAST_YEAR:-20}
RUNS_DIRECTORY=${RUNS_DIRECTORY:-$HOME/catke_diffusivity_sensitivity}
FIGURE_DIRECTORY=${FIGURE_DIRECTORY:-$RUNS_DIRECTORY/figures}
PROJECT=${PROJECT:-.}

source $HOME/env_nvhpc_26.5.sh
export JULIA_NUM_THREADS=32

julia +1.12 --project=$PROJECT examples/CATKE_diffusivity_sensitivity/plot_catke_diffusivity.jl \
    --runs-directory $RUNS_DIRECTORY --figure-directory $FIGURE_DIRECTORY \
    --first-year $FIRST_YEAR --last-year $LAST_YEAR

# Buoyancy, isopycnal slopes and overturning, drawn as for the Gent-McWilliams tapering runs
julia +1.12 --project=$PROJECT examples/GM_tapering_sensitivity/plot_gm_tapering.jl \
    --runs-directory $RUNS_DIRECTORY --figure-directory $FIGURE_DIRECTORY/tapering_figures \
    --first-year $FIRST_YEAR --last-year $LAST_YEAR \
    --run-names catke_0.5 catke_1.0 catke_2.0 --run-labels "κ × 0.5" "control" "κ × 2" \
    --reference-run catke_1.0 --max-slopes 0.01 0.01 0.01
