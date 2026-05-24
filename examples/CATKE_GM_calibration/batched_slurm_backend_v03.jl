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
isdefined(ClimaCalibrate, :Backend) || error(
    "batched_slurm_backend_v03.jl requires ClimaCalibrate v0.3.x; " *
    "the active project appears to resolve an older ClimaCalibrate."
)
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
import TOML
import Dates

const GPUS_PER_NODE_V03 = 8
const THREADS_PER_MEMBER_V03 = 12

# ----------------------------------------------------------------------
# Pre-emption-safe state: persistent per-iteration ledger of submitted
# Slurm job IDs + sacct-based liveness check + extended checkpoint states.
#
# When the monitor (driver) Slurm job is pre-empted and Slurm requeues it,
# the new driver process must NOT re-submit forward-model batches that are
# still alive under their original job_id. The ledger records, for each
# ensemble member, the Slurm job_id that covers it. `job_is_alive` consults
# sacct (which, unlike squeue, survives pre-emption/requeue cycles and
# reports terminal states for finished jobs).
# ----------------------------------------------------------------------

const LIVE_SACCT_STATES = Set([
    "PENDING", "RUNNING", "REQUEUED", "RESIZING",
    "SUSPENDED", "CONFIGURING", "COMPLETING",
])

function job_is_alive(job_id::AbstractString)
    isempty(job_id) && return false
    out = try
        read(`sacct -j $job_id -X -n -P -o State`, String)
    catch err
        @warn "sacct query failed for job $job_id; treating as alive to be safe" exception=err
        return true
    end
    states = filter(!isempty, strip.(split(out, '\n')))
    isempty(states) && return false
    return any(s -> first(split(s, ' ')) in LIVE_SACCT_STATES, states)
end

ledger_path(output_dir, iter) =
    joinpath(path_to_iteration(output_dir, iter), "job_ledger.toml")

function load_ledger(output_dir, iter)
    file = ledger_path(output_dir, iter)
    ledger = Dict{Int, NamedTuple{(:job_id, :batch_idx, :submitted_at),
                                  Tuple{String, Int, String}}}()
    isfile(file) || return ledger
    data = try
        TOML.parsefile(file)
    catch err
        @warn "Failed to parse ledger $file; ignoring" exception=err
        return ledger
    end
    members = get(data, "members", Dict{String,Any}())
    for (k, v) in members
        member = parse(Int, k)
        ledger[member] = (
            job_id       = String(get(v, "job_id", "")),
            batch_idx    = Int(get(v, "batch_idx", 0)),
            submitted_at = String(get(v, "submitted_at", "")),
        )
    end
    return ledger
end

function update_ledger!(output_dir, iter, members::Vector{Int},
                        job_id::AbstractString, batch_idx::Integer)
    file = ledger_path(output_dir, iter)
    mkpath(dirname(file))
    ledger = load_ledger(output_dir, iter)
    submitted_at = string(Dates.now())
    for m in members
        ledger[m] = (job_id = String(job_id),
                     batch_idx = Int(batch_idx),
                     submitted_at = submitted_at)
    end
    data = Dict("members" => Dict(string(m) => Dict(
        "job_id"       => e.job_id,
        "batch_idx"    => e.batch_idx,
        "submitted_at" => e.submitted_at,
    ) for (m, e) in ledger))
    tmp = file * ".tmp"
    open(tmp, "w") do io
        TOML.print(io, data)
    end
    mv(tmp, file; force = true)
    return nothing
end

function member_checkpoint_status(output_dir, iter, member)
    file = checkpoint_path(output_dir, iter, member)
    isfile(file) || return :missing
    s = try
        strip(readline(file))
    catch
        return :missing
    end
    s == "completed"   && return :completed
    s == "failed"      && return :failed
    s == "in_progress" && return :in_progress
    return :unknown
end

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
                                    poll_interval = 30, timeout = nothing,
                                    sacct_check_interval = 300)
    isempty(members) && return

    @info "Waiting for $(length(members)) ensemble member(s) to finish: $members"

    start_time = time()
    finished_members = Set{Int}()
    last_sacct_check = 0.0

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
            elapsed = time() - start_time

            if !isnothing(timeout) && elapsed > timeout
                not_finished = setdiff(Set(members), finished_members)
                error("Timeout waiting for members to finish after $(timeout)s. " *
                      "Members not finished: $not_finished")
            end

            if elapsed > 0 && mod(round(Int, elapsed), 1800) < poll_interval
                not_done = setdiff(Set(members), finished_members)
                @info "Still waiting for members: $not_done (elapsed: $(round(elapsed/60, digits=1)) min)"
            end

            # Soft job-died guard: every sacct_check_interval seconds, warn
            # about any unfinished member whose Slurm job is now in a terminal
            # sacct state. We do NOT auto-resubmit here — the next driver
            # restart will re-classify and re-submit if needed. This is just
            # an operator signal that requeue isn't happening.
            if elapsed - last_sacct_check >= sacct_check_interval
                last_sacct_check = elapsed
                ledger = load_ledger(output_dir, iter)
                for member in members
                    member in finished_members && continue
                    entry = get(ledger, member, nothing)
                    entry === nothing && continue
                    if !job_is_alive(entry.job_id)
                        @warn "Member $member: Slurm job $(entry.job_id) appears terminated, " *
                              "but checkpoint not yet completed/failed. Awaiting Slurm requeue " *
                              "or manual intervention."
                    end
                end
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
        gpu_id          = gpu_idx - 1
        member_log      = path_to_model_log(output_dir, iter, member)
        member_path     = path_to_ensemble_member(output_dir, iter, member)
        member_ckpt     = checkpoint_path(output_dir, iter, member)

        cmd = """
# Member $member on GPU $gpu_id
(
    mkdir -p "$member_path"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    # Pre-empt+requeue safety: if this member already finished in a previous
    # invocation of this batch (Slurm restarted the whole job under the same
    # job_id), skip rather than restart from t=0.
    if [ -f "$member_ckpt" ]; then
        status=\$(cat "$member_ckpt" 2>/dev/null | head -n1)
        if [ "\$status" = "completed" ] || [ "\$status" = "failed" ]; then
            echo "[Member $member] Already \$status; skipping"
            exit 0
        fi
    fi
    mkdir -p "\$(dirname "$member_ckpt")"
    echo "in_progress" > "$member_ckpt"
    echo "[Member $member] Starting on GPU $gpu_id at \$(date)"
    script -q -c "julia +1.12.3 --threads=$(THREADS_PER_MEMBER_V03) $exeflags --project=$experiment_dir -e \\\"
        import ClimaCalibrate as CAL
        iteration = $iter
        member = $member
        output_dir = \\\\\\\"$output_dir\\\\\\\"
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

    iter_path = path_to_iteration(output_dir, iter)
    mkpath(iter_path)

    # Pre-emption-safe classification: on every entry (including driver
    # restart after monitor pre-emption), decide per member whether it is
    # already terminal, already covered by a live Slurm job, or needs a
    # fresh submission.
    ledger        = load_ledger(output_dir, iter)
    alive_members = Int[]
    needs_submit  = Int[]

    for member in 1:ensemble_size
        status = member_checkpoint_status(output_dir, iter, member)
        if status === :completed
            @info "Skipping completed member $member"
            continue
        elseif status === :failed
            @info "Skipping previously-failed member $member (will not retry)"
            continue
        end

        # status is :missing, :in_progress, or :unknown
        entry = get(ledger, member, nothing)
        if entry !== nothing && job_is_alive(entry.job_id)
            @info "Member $member already covered by live job $(entry.job_id) (status=$status); not resubmitting"
            push!(alive_members, member)
        else
            if entry !== nothing
                @info "Member $member: prior job $(entry.job_id) is terminal/unknown (status=$status); resubmitting"
            end
            push!(needs_submit, member)
        end
    end

    if isempty(alive_members) && isempty(needs_submit)
        @info "All members already completed for iteration $iter"
        return nothing
    end

    if !isempty(needs_submit)
        n_batches = ceil(Int, length(needs_submit) / backend.gpus_per_node)
        @info "Submitting $n_batches batched job(s) for $(length(needs_submit)) members " *
              "($(length(alive_members)) members already covered by live jobs)"

        for batch_idx in 1:n_batches
            start_idx     = (batch_idx - 1) * backend.gpus_per_node + 1
            end_idx       = min(batch_idx * backend.gpus_per_node, length(needs_submit))
            batch_members = needs_submit[start_idx:end_idx]
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
            update_ledger!(output_dir, iter, batch_members, string(job_info.id), batch_idx)
        end
    else
        @info "No new submissions needed; $(length(alive_members)) members still under live jobs"
    end

    # No outer timeout: with Slurm requeue, pre-empted jobs may sit PENDING
    # arbitrarily long. Monitor wall-time is the real upper bound.
    wait_members = vcat(alive_members, needs_submit)
    wait_for_member_completion(output_dir, iter, wait_members;
                               poll_interval = 30, timeout = nothing)
    return nothing
end
