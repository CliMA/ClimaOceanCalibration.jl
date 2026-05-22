# batched_slurm_backend_v03.jl
#
# v0.3.0-compatible port of batched_slurm_backend.jl.
# Submits ONE Slurm job per batch of `gpus_per_node` ensemble members, forking
# `n_members` Julia subprocesses inside a single allocation, each pinned to its
# own GPU via CUDA_VISIBLE_DEVICES. Polling is done on per-member checkpoint
# files (more reliable than squeue when the batch job wraps many members).
#
# Designed against ClimaCalibrate v0.3.0. Extension points:
#   * ClimaCalibrate.Backend.module_load_string(::BatchedSlurmGCPBackendV03)
#   * ClimaCalibrate.Backend.make_job_script(::BatchedSlurmGCPBackendV03, body; ...)
#   * ClimaCalibrate.Calibration.run_iteration(backend::BatchedSlurmGCPBackendV03, ...)

import ClimaCalibrate
import ClimaCalibrate: Backend, Calibration
import ClimaCalibrate.Backend:
    HPCBackend, SlurmBackend, SlurmConfig, JobInfo,
    generate_directives, generate_env_vars
import ClimaCalibrate.Calibration:
    checkpoint_path,
    path_to_iteration,
    path_to_ensemble_member,
    path_to_model_log,
    model_completed,
    model_started,
    write_model_started

const GPUS_PER_NODE_V03 = 8

"""
    BatchedSlurmGCPBackendV03 <: SlurmBackend

A v0.3.0 SlurmBackend that submits one Slurm job per batch of
`gpus_per_node` ensemble members and runs them in parallel on one node.
"""
struct BatchedSlurmGCPBackendV03 <: SlurmBackend
    hpc_config::SlurmConfig
    job_records::Vector{JobInfo}
    gpus_per_node::Int
end

function BatchedSlurmGCPBackendV03(config::SlurmConfig; gpus_per_node = GPUS_PER_NODE_V03)
    return BatchedSlurmGCPBackendV03(config, JobInfo[], gpus_per_node)
end

# ----------------------------------------------------------------------
# Cluster environment (a3mega, NVHPC 25.7, HPC-X 2.22.1, Julia 1.12.3)
# ----------------------------------------------------------------------
function ClimaCalibrate.Backend.module_load_string(::BatchedSlurmGCPBackendV03)
    return """
# Load NVHPC 25.7 environment (CUDA 12.9, HPC-X 2.22.1)
source \$HOME/env_nvhpc_25.7.sh

# Julia and CUDA Settings
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NVTX_CALLBACKS=gc

# Load API keys
source ~/API_keys.sh

# Navigate to Project Directory
cd ~/CES_oceananigans/ClimaOceanCalibration.jl

echo "Environment:"
echo "  NVHPC Version: 25.7"
echo "  CUDA Version: 12.9"
echo "  CUDA_HOME: \$CUDA_HOME"
echo "  MPI_HOME: \$MPI_HOME"
echo "  Julia: \$(julia +1.12.3 --version)"
"""
end

# Override make_job_script: the default SlurmBackend version wraps the body in
# `srun --output=...`, which is wrong for our forked-subprocess pattern. We
# emit the script frame ourselves but reuse generate_directives / generate_env_vars /
# module_load_string from the v0.3.0 API.
function ClimaCalibrate.Backend.make_job_script(
    backend::BatchedSlurmGCPBackendV03,
    job_body;
    job_name = "batched_slurm_job",
    output = "output.txt",
)
    (; hpc_config) = backend
    return """
#!/bin/bash
$(generate_directives(hpc_config))
#SBATCH --job-name=$job_name
#SBATCH --output=$output

$(ClimaCalibrate.Backend.module_load_string(backend))
$(generate_env_vars(hpc_config))

$job_body
exit 0
"""
end

# ----------------------------------------------------------------------
# Per-member checkpoint-file polling (ported verbatim from the v0.1 backend)
# ----------------------------------------------------------------------
write_model_failed(output_dir, iteration, member) =
    open(checkpoint_path(output_dir, iteration, member), "w") do io
        write(io, "failed")
    end

function model_finished(output_dir, iteration, member)
    file = checkpoint_path(output_dir, iteration, member)
    !isfile(file) && return false
    status = readline(file)
    return status == "completed" || status == "failed"
end

function wait_for_member_completion(output_dir, iter, members::Vector{Int};
                                    poll_interval = 30, timeout = nothing)
    isempty(members) && return

    @info "Waiting for $(length(members)) ensemble member(s) to finish: $members"

    start_time = time()
    finished_members = Set{Int}()

    while length(finished_members) < length(members)
        for member in members
            member in finished_members && continue
            if model_finished(output_dir, iter, member)
                push!(finished_members, member)
                file = checkpoint_path(output_dir, iter, member)
                status = readline(file)
                status_str = status == "completed" ? "completed" : "FAILED"
                @info "Member $member $status_str ($(length(finished_members))/$(length(members)) done)"
            end
        end

        if length(finished_members) < length(members)
            if !isnothing(timeout)
                elapsed = time() - start_time
                if elapsed > timeout
                    not_finished = setdiff(Set(members), finished_members)
                    error("Timeout waiting for members to finish after $(timeout)s. " *
                          "Members not finished: $not_finished")
                end
            end

            elapsed = time() - start_time
            if elapsed > 0 && mod(round(Int, elapsed), 1800) < poll_interval
                not_done = setdiff(Set(members), finished_members)
                @info "Still waiting for members: $not_done (elapsed: $(round(elapsed/60, digits=1)) min)"
            end

            sleep(poll_interval)
        end
    end

    @info "All $(length(members)) ensemble members have finished"
end

# ----------------------------------------------------------------------
# Build the per-batch job body (the forked-subprocess block + wait loop).
# This is just the body, NOT the whole sbatch script — make_job_script
# wraps it with #SBATCH headers, module loads, and env vars.
# ----------------------------------------------------------------------
function batched_job_body(
    iter::Integer,
    members::Vector{Int},
    output_dir,
    model_interface_filepath,
    experiment_dir,
    exeflags,
)
    n_gpus = length(members)
    interface_jld2 = joinpath(output_dir, "interface.jld2")

    launch_blocks = String[]
    for (gpu_idx, member) in enumerate(members)
        gpu_id      = gpu_idx - 1
        member_log  = path_to_model_log(output_dir, iter, member)
        member_path = path_to_ensemble_member(output_dir, iter, member)

        cmd = """
# Member $member on GPU $gpu_id
(
    mkdir -p "$member_path"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "[Member $member] Starting on GPU $gpu_id at \$(date)"
    script -q -c "julia +1.12.3 $exeflags --project=$experiment_dir -e \\\"
        import ClimaCalibrate as CAL
        iteration = $iter
        member = $member
        model_interface_filepath = \\\\\\\"$model_interface_filepath\\\\\\\"
        include(model_interface_filepath)
        interface = CAL._load(\\\\\\\"$interface_jld2\\\\\\\")
        try
            CAL.forward_model(interface, iteration, member)
            CAL.write_model_completed(\\\\\\\"$output_dir\\\\\\\", iteration, member)
            println(\\\\\\\"[Member $member] Completed successfully\\\\\\\")
        catch e
            println(\\\\\\\"[Member $member] FAILED with error:\\\\\\\")
            showerror(stdout, e, catch_backtrace())
            open(CAL.checkpoint_path(\\\\\\\"$output_dir\\\\\\\", iteration, member), \\\\\\\"w\\\\\\\") do io
                write(io, \\\\\\\"failed\\\\\\\")
            end
            rethrow(e)
        end
    \\\"" /dev/null >> "$member_log" 2>&1
    echo "[Member $member] Finished at \$(date)"
) &
PIDS[$gpu_id]=\$!
"""
        push!(launch_blocks, cmd)
    end

    max_idx    = n_gpus - 1
    wait_block = """
# Wait for all background processes and collect exit codes
echo "Waiting for all $n_gpus members to complete..."
FAILED=0
for i in \$(seq 0 $max_idx); do
    wait \${PIDS[\$i]}
    EXIT_CODE=\$?
    if [ \$EXIT_CODE -ne 0 ]; then
        echo "Process \$i (PID \${PIDS[\$i]}) failed with exit code \$EXIT_CODE"
        FAILED=\$((FAILED + 1))
    fi
done

if [ \$FAILED -gt 0 ]; then
    echo "WARNING: \$FAILED member(s) failed"
    exit 1
fi

echo "All members completed successfully"
"""

    return """
echo "========================================"
echo "Batched Job: Iteration $iter"
echo "Members: $(join(members, ", "))"
echo "GPUs requested: $n_gpus"
echo "Started at: \$(date)"
echo "========================================"

# Array to store PIDs
declare -a PIDS

$(join(launch_blocks, "\n"))

$wait_block

echo "========================================"
echo "Batch completed at: \$(date)"
echo "========================================"
"""
end

# ----------------------------------------------------------------------
# Build a per-batch backend whose SlurmConfig has the right directives
# (nodes=1, ntasks=n_gpus, gres=gpu:n_gpus) overlaid on the shared config.
# Uses the public SlurmConfig(; directives, modules, env_vars) constructor
# so we benefit from time formatting, GPU/CPU CLIMACOMMS_DEVICE inference,
# and uniqueness checks.
# ----------------------------------------------------------------------
function batch_backend(parent::BatchedSlurmGCPBackendV03, n_gpus::Integer)
    shared = parent.hpc_config
    directives = collect(shared.directives)  # Vector{Pair{Symbol,Any}}
    # Drop directives we're about to override.
    overridden = Set([:nodes, :ntasks, :gres, :gpus_per_task, :ntasks_per_node, :gpus])
    directives = filter(p -> !(first(p) in overridden), directives)
    push!(directives, :nodes  => 1)
    push!(directives, :ntasks => n_gpus)
    push!(directives, :gres   => "gpu:$n_gpus")

    env_vars = collect(shared.env_vars)  # Vector{Pair{String,Any}}

    batch_cfg = SlurmConfig(;
        directives = directives,
        modules    = copy(shared.modules),
        env_vars   = env_vars,
    )
    return BatchedSlurmGCPBackendV03(batch_cfg, parent.job_records, parent.gpus_per_node)
end

# ----------------------------------------------------------------------
# Override Calibration.run_iteration for our batched backend.
# v0.3.0 calibrate() calls run_iteration with this exact signature:
#   run_iteration(backend, iter, ensemble_size, output_dir,
#                 model_interface_filepath, experiment_dir, exeflags)
# ----------------------------------------------------------------------
function ClimaCalibrate.Calibration.run_iteration(
    backend::BatchedSlurmGCPBackendV03,
    iter,
    ensemble_size,
    output_dir,
    model_interface_filepath,
    experiment_dir,
    exeflags,
)
    @info "Iteration $iter — batched submission for $ensemble_size members " *
          "($(backend.gpus_per_node) per node)"

    members_to_run = Int[]
    for member in 1:ensemble_size
        if model_completed(output_dir, iter, member)
            @info "Skipping completed member $member"
        else
            push!(members_to_run, member)
        end
    end

    if isempty(members_to_run)
        @info "All members already completed for iteration $iter"
        return nothing
    end

    n_batches = ceil(Int, length(members_to_run) / backend.gpus_per_node)
    @info "Submitting $n_batches batched job(s) for $(length(members_to_run)) members"

    iter_path = path_to_iteration(output_dir, iter)
    mkpath(iter_path)

    for batch_idx in 1:n_batches
        start_idx     = (batch_idx - 1) * backend.gpus_per_node + 1
        end_idx       = min(batch_idx * backend.gpus_per_node, length(members_to_run))
        batch_members = members_to_run[start_idx:end_idx]
        n_gpus        = length(batch_members)
        @info "Batch $batch_idx: members $batch_members"

        for member in batch_members
            write_model_started(output_dir, iter, member)
        end

        body = batched_job_body(iter, batch_members, output_dir,
                                model_interface_filepath, experiment_dir, exeflags)
        bbackend = batch_backend(backend, n_gpus)
        script_str = Backend.make_job_script(
            bbackend, body;
            job_name = "iter$(iter)_batch$(batch_idx)",
            output   = joinpath(iter_path, "batch_$(batch_idx).log"),
        )

        sbatch_filepath = joinpath(iter_path, "batch_$(batch_idx).sbatch")
        write(sbatch_filepath, script_str)
        @info "Wrote sbatch script to: $sbatch_filepath"

        job_info = Backend.submit_job(backend, script_str)
        @info "Submitted batched job $(job_info.id) for iteration $iter batch $batch_idx (members: $batch_members)"
    end

    # 7-day safety net. Completion markers are written on both success and failure.
    wait_for_member_completion(output_dir, iter, members_to_run;
                               poll_interval = 30, timeout = 604800)
    return nothing
end
