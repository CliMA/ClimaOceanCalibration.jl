# batched_slurm_backend.jl
# A Slurm backend that submits a single job for multiple ensemble members,
# utilizing all GPUs on the requested nodes efficiently.
#
# On GCP a3mega nodes, each node has 8 GPUs. This backend submits one job
# per node-batch of ensemble members, running up to 8 forward models in parallel
# on a single node.

using ClimaCalibrate
using ClimaCalibrate: HPCBackend, path_to_iteration, path_to_ensemble_member,
                      path_to_model_log, write_model_started, write_model_completed,
                      model_completed, model_started,
                      generate_sbatch_directives, submit_slurm_job
using EnsembleKalmanProcesses: EnsembleKalmanProcess

using ClimaCalibrate: checkpoint_path

import ClimaCalibrate: run_hpc_iteration, module_load_string

"""
    write_model_failed(output_dir, iteration, member)

Write a "failed" marker to the checkpoint file for a failed ensemble member.
"""
write_model_failed(output_dir, iteration, member) =
    open(checkpoint_path(output_dir, iteration, member), "w") do io
        write(io, "failed")
    end

"""
    model_finished(output_dir, iteration, member) -> Bool

Check if a model run has finished (either completed successfully or failed).
Returns true if the checkpoint file contains "completed" or "failed".
"""
function model_finished(output_dir, iteration, member)
    file = checkpoint_path(output_dir, iteration, member)
    !isfile(file) && return false
    status = readline(file)
    return status == "completed" || status == "failed"
end

"""
    get_slurm_job_status(job_id::Int) -> Symbol

Query Slurm for the status of a job. Returns one of:
- :PENDING - job is waiting to run
- :RUNNING - job is currently running
- :COMPLETED - job finished successfully
- :FAILED - job failed or was cancelled
- :UNKNOWN - job not found (may not be registered yet)
"""
function get_slurm_job_status(job_id::Int)
    cmd = `squeue -j $job_id --format=%T --noheader`
    stdout_pipe = Pipe()
    stderr_pipe = Pipe()
    process = run(pipeline(ignorestatus(cmd), stdout=stdout_pipe, stderr=stderr_pipe))
    close(stdout_pipe.in)
    close(stderr_pipe.in)

    status_str = strip(String(read(stdout_pipe)))
    stderr_str = String(read(stderr_pipe))
    exit_code = process.exitcode

    # Job not in queue - could be completed or never existed
    if status_str == "" && exit_code == 0 && stderr_str == ""
        return :COMPLETED
    end

    # Invalid job ID error - job finished and left accounting
    if exit_code != 0 && contains(stderr_str, "Invalid job id")
        return :COMPLETED
    end

    # Check for various Slurm states
    pending_states = ["PENDING", "CONFIGURING", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED", "RESIZING"]
    running_states = ["RUNNING", "COMPLETING", "STAGED", "SUSPENDED", "STOPPED"]
    failed_states = ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"]

    for state in pending_states
        contains(status_str, state) && return :PENDING
    end
    for state in running_states
        contains(status_str, state) && return :RUNNING
    end
    for state in failed_states
        contains(status_str, state) && return :FAILED
    end

    # Unknown status
    @warn "Job $job_id has unknown status: '$status_str'"
    return :UNKNOWN
end

"""
    wait_for_member_completion(output_dir, iter, members; poll_interval=30, timeout=nothing)

Wait for all ensemble members to finish (either success or failure).

This is more reliable than checking Slurm job status because:
1. The marker is written by Julia AFTER all file I/O is done
2. It doesn't depend on Slurm's job accounting being consistent

Arguments:
- `output_dir`: Calibration output directory
- `iter`: Current iteration number
- `members`: Vector of member numbers to wait for
- `poll_interval`: Seconds between checks (default: 30)
- `timeout`: Max seconds to wait, or nothing for no timeout (default: nothing)
"""
function wait_for_member_completion(output_dir, iter, members::Vector{Int}; poll_interval=30, timeout=nothing)
    if isempty(members)
        return
    end

    @info "Waiting for $(length(members)) ensemble member(s) to finish: $members"

    start_time = time()
    finished_members = Set{Int}()

    while length(finished_members) < length(members)
        for member in members
            member in finished_members && continue

            if model_finished(output_dir, iter, member)
                push!(finished_members, member)
                # Check if it was success or failure
                file = checkpoint_path(output_dir, iter, member)
                status = readline(file)
                status_str = status == "completed" ? "completed" : "FAILED"
                @info "Member $member $status_str ($(length(finished_members))/$(length(members)) done)"
            end
        end

        if length(finished_members) < length(members)
            # Check timeout
            if !isnothing(timeout)
                elapsed = time() - start_time
                if elapsed > timeout
                    not_finished = setdiff(Set(members), finished_members)
                    error("Timeout waiting for members to finish after $(timeout)s. " *
                          "Members not finished: $not_finished")
                end
            end

            # Log progress periodically
            elapsed = time() - start_time
            if elapsed > 0 && mod(round(Int, elapsed), 1800) < poll_interval  # Every ~5 minutes
                not_done = setdiff(Set(members), finished_members)
                @info "Still waiting for members: $not_done (elapsed: $(round(elapsed/60, digits=1)) minutes)"
            end

            sleep(poll_interval)
        end
    end

    @info "All $(length(members)) ensemble members have finished"
end

"""
    BatchedSlurmGCPBackend <: HPCBackend

A backend that submits a single Slurm job per node, running multiple ensemble
members in parallel (one per GPU). This efficiently utilizes all GPUs on GCP
a3mega nodes where each node has 8 GPUs.

# Configuration
- `gpus_per_node`: Number of GPUs per node (default: 8 for GCP a3mega)

# Usage
```julia
include("batched_slurm_backend.jl")

hpc_kwargs = Dict(
    :time => 120,           # minutes
    :partition => "a3mega",
    :cpus_per_task => 4,
    :mem => "128G",
)

ekp = ClimaCalibrate.calibrate(
    BatchedSlurmGCPBackend,
    ekp,
    n_iterations,
    prior,
    output_dir;
    model_interface = model_interface,
    hpc_kwargs = hpc_kwargs,
)
```

# Notes
- Each forward model runs on exactly 1 GPU
- Ensemble members are batched into groups of `gpus_per_node` (default 8)
- If ensemble_size > gpus_per_node, multiple batched jobs are submitted
"""
struct BatchedSlurmGCPBackend <: HPCBackend end

# Number of GPUs per node on GCP a3mega
const GPUS_PER_NODE = 8

function module_load_string(::Type{BatchedSlurmGCPBackend})
    # Note: CUDA_VISIBLE_DEVICES is set per-process in the batch script, not here
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

ClimaCalibrate.backend_worker_kwargs(::Type{BatchedSlurmGCPBackend}) = (; partition = "a3mega")

"""
Override run_hpc_iteration to batch ensemble members into single jobs.
"""
function run_hpc_iteration(
    ::Type{BatchedSlurmGCPBackend},
    ekp::EnsembleKalmanProcess,
    iter,
    ensemble_size,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str,
    prior;
    hpc_kwargs,
    verbose = false,
    exeflags = "",
)
    @info "Iteration $iter - Batched submission for $ensemble_size members"

    # Identify members that need to run (skip completed ones)
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
        return
    end

    # Batch members into groups of GPUS_PER_NODE
    n_batches = ceil(Int, length(members_to_run) / GPUS_PER_NODE)
    @info "Submitting $n_batches batched job(s) for $(length(members_to_run)) members"

    job_ids = Int[]  # Must be Int for Slurm (String triggers PBS code path)
    for batch_idx in 1:n_batches
        start_idx = (batch_idx - 1) * GPUS_PER_NODE + 1
        end_idx = min(batch_idx * GPUS_PER_NODE, length(members_to_run))
        batch_members = members_to_run[start_idx:end_idx]

        @info "Batch $batch_idx: members $(batch_members)"

        job_id = submit_batched_job(
            BatchedSlurmGCPBackend,
            iter,
            batch_idx,
            batch_members,
            output_dir,
            experiment_dir,
            model_interface,
            module_load_str;
            hpc_kwargs,
            exeflags,
        )

        if !isnothing(job_id)
            push!(job_ids, job_id)
        end
    end

    # Wait for all ensemble members to write their completion markers
    # This is more reliable than checking Slurm job status because the marker
    # is written by Julia AFTER all simulation output is complete
    #
    # Timeout is set to 7 days (604800 seconds) as a safety net for very long runs
    # The completion marker is written even on failure, so this should always complete
    if !isempty(members_to_run)
        wait_for_member_completion(output_dir, iter, members_to_run;
                                   poll_interval=30,
                                   timeout=604800)  # 7 days
    end
end

"""
Submit a single Slurm job that runs multiple ensemble members in parallel.
"""
function submit_batched_job(
    ::Type{BatchedSlurmGCPBackend},
    iter,
    batch_idx,
    members,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
    exeflags = "",
)
    # Mark all members as started
    for member in members
        write_model_started(output_dir, iter, member)
    end

    # Generate the batched sbatch script
    sbatch_contents = generate_batched_sbatch_script(
        iter,
        batch_idx,
        members,
        output_dir,
        experiment_dir,
        model_interface,
        module_load_str;
        hpc_kwargs,
        exeflags,
    )

    # Write script to a permanent file for debugging
    iter_path = path_to_iteration(output_dir, iter)
    mkpath(iter_path)

    sbatch_filepath = joinpath(iter_path, "batch_$(batch_idx).sbatch")
    write(sbatch_filepath, sbatch_contents)
    @info "Wrote sbatch script to: $sbatch_filepath"

    # Submit with error capture
    job_id = try
        submit_slurm_job(sbatch_filepath)
    catch e
        # Try to get more detailed error from sbatch
        @error "sbatch submission failed. Attempting to get detailed error..."
        try
            result = read(`sbatch --parsable $sbatch_filepath`, String)
            @info "sbatch output: $result"
        catch e2
            err_output = try
                read(pipeline(`sbatch $sbatch_filepath`, stderr=stdout), String)
            catch e3
                "Could not capture error: $e3"
            end
            @error "sbatch error output: $err_output"
        end
        rethrow(e)
    end

    @info "Submitted batched job $job_id for iteration $iter batch $batch_idx (members: $members)"

    return job_id
end

"""
Generate the sbatch script content for a batched job.
"""
function generate_batched_sbatch_script(
    iter,
    batch_idx,
    members,
    output_dir,
    experiment_dir,
    model_interface,
    module_load_str;
    hpc_kwargs,
    exeflags = "",
)
    n_gpus = length(members)

    # Modify hpc_kwargs for batched submission
    # Request one node with n_gpus GPUs using --gres=gpu:N format
    batch_kwargs = copy(hpc_kwargs)
    batch_kwargs[:nodes] = 1
    batch_kwargs[:ntasks] = n_gpus
    batch_kwargs[:gres] = "gpu:$n_gpus"  # Standard Slurm GPU allocation
    # Remove conflicting keys
    delete!(batch_kwargs, :gpus_per_task)
    delete!(batch_kwargs, :ntasks_per_node)
    delete!(batch_kwargs, :gpus)

    # Handle boolean flags separately (Slurm wants --exclusive, not --exclusive=true)
    boolean_flags = String[]
    if get(batch_kwargs, :exclusive, false)
        push!(boolean_flags, "#SBATCH --exclusive")
        delete!(batch_kwargs, :exclusive)
    end

    slurm_directives = generate_sbatch_directives(batch_kwargs)

    # Append boolean flags
    if !isempty(boolean_flags)
        slurm_directives = slurm_directives * "\n" * join(boolean_flags, "\n")
    end

    # Main log for the batch
    iter_path = path_to_iteration(output_dir, iter)
    batch_log = joinpath(iter_path, "batch_$(batch_idx).log")

    # Generate launch commands for each member
    # Each member gets assigned to a specific GPU via CUDA_VISIBLE_DEVICES
    launch_commands = String[]
    for (gpu_idx, member) in enumerate(members)
        gpu_id = gpu_idx - 1  # 0-indexed GPU IDs
        member_log = path_to_model_log(output_dir, iter, member)

        # Ensure member directory exists
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
        output_dir = \\\\\\\"$output_dir\\\\\\\"
        model_interface = \\\\\\\"$model_interface\\\\\\\"
        include(model_interface)
        try
            CAL.forward_model(iteration, member)
            CAL.write_model_completed(\\\\\\\"$output_dir\\\\\\\", iteration, member)
            println(\\\\\\\"[Member $member] Completed successfully\\\\\\\")
        catch e
            println(\\\\\\\"[Member $member] FAILED with error:\\\\\\\")
            showerror(stdout, e, catch_backtrace())
            # Write failed marker so wait doesn't hang
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
        push!(launch_commands, cmd)
    end

    all_launch_commands = join(launch_commands, "\n")

    # Generate wait and status check commands
    n_members = length(members)
    max_idx = n_members - 1
    wait_commands = """
# Wait for all background processes and collect exit codes
echo "Waiting for all $n_members members to complete..."
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

    sbatch_contents = """#!/bin/bash
#SBATCH --job-name=iter$(iter)_batch$(batch_idx)
#SBATCH --output=$batch_log
$slurm_directives

echo "========================================"
echo "Batched Job: Iteration $iter, Batch $batch_idx"
echo "Members: $(join(members, ", "))"
echo "GPUs requested: $n_gpus"
echo "Started at: \$(date)"
echo "========================================"

$module_load_str

# Array to store PIDs
declare -a PIDS

$all_launch_commands

$wait_commands

echo "========================================"
echo "Batch $batch_idx completed at: \$(date)"
echo "========================================"
exit 0
"""

    return sbatch_contents
end
