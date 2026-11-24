#!/bin/bash
#SBATCH --partition=a3mega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --job-name=quarter_catke
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

# sbatch --export=ALL,DIFFUSIVITY_SCALES="0.5 1.0 2.0" examples/CATKE_diffusivity_sensitivity/run_quarter_degree_catke.slurm  (from the repository root)
# One run per GPU of the node. A run writes its three-dimensional output from FIELD_OUTPUT_START_DAY on. A requeued
# job continues every run from its latest checkpoint.

DIFFUSIVITY_SCALES=${DIFFUSIVITY_SCALES:-0.5 1.0 2.0}
STOP_DAY=${STOP_DAY:-7300}
FIELD_OUTPUT_START_DAY=${FIELD_OUTPUT_START_DAY:-5040}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-$HOME/quarter_degree_catke}
PROJECT=${PROJECT:-.}
RUN_OPTIONS=${RUN_OPTIONS:-}

source $HOME/env_nvhpc_26.5.sh
source $HOME/API_keys.sh
unset JULIA_CUDA_MEMORY_POOL
export JULIA_NUM_THREADS=16

mkdir -p logs
julia +1.12 --project=$PROJECT -e 'using ClimaOcean, Oceananigans, CUDA'

function run_one {
    scale=$1
    export CUDA_VISIBLE_DEVICES=$2
    directory=$OUTPUT_DIRECTORY/quarter_catke_$scale
    log=logs/${SLURM_JOB_NAME}_scale_$scale.out
    run="julia +1.12 --project=$PROJECT examples/CATKE_diffusivity_sensitivity/run_quarter_degree_catke.jl \
         --diffusivity-scale $scale --output-directory $OUTPUT_DIRECTORY $RUN_OPTIONS"
    mkdir -p $directory

    if [ ! -f $directory/field_output_started ]; then
        $run --stop-day $FIELD_OUTPUT_START_DAY >> $log 2>&1 || return 1
        touch $directory/field_output_started
    fi

    $run --stop-day $STOP_DAY --field-output >> $log 2>&1
}

gpu=0
runs=""
for scale in $DIFFUSIVITY_SCALES; do
    run_one $scale $gpu &
    runs="$runs $!"
    gpu=$((gpu + 1))
done

status=0
for run in $runs; do
    wait $run || status=1
done
exit $status
