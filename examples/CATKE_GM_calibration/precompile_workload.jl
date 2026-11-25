# precompile_workload.jl
#
# PackageCompiler precompile-execution trace for the CATKE+GM ORCA calibration
# forward model. This script is passed to `create_sysimage(...;
# precompile_execution_file = ...)` by build_sysimage.jl. PackageCompiler runs
# it and records every method that gets compiled, baking that native code into
# the sysimage so each cold per-member calibration process starts hot.
#
# It runs the production forward model, run_CATKE_GM_calibration_orca in
# forward_model_orca.jl, for ten time steps instead of years, so the same
# GPU-typed method specializations are traced.
#
# Requirements (true on a3mega H100 nodes):
#   - a functional GPU (the workload uses arch = GPU())
#   - the real JRA55 / WOA-ECCO / ORCA bathymetry input data, with the JRA55
#     forcing located via the FORCING_DIR env var (default matches forward_model_orca.jl)
#
# A failure here is non-fatal to the sysimage build: build_sysimage.jl still
# bakes everything compiled up to the point of failure. We therefore wrap the
# body in try/catch and only log.

using Oceananigans.Units

include(joinpath(@__DIR__, "forward_model_orca.jl"))

# GM on and off are different model types, so both are traced into the one sysimage.
function _precompile_workload(use_gm::Bool)
    trace_length = 10 * OneDegree().time_step.initial_Δt / 365days   # years
    config_dict = Dict{String,Any}("output_dir"        => mktempdir(; prefix = "sysimage_trace_"),
                                   "forcing_dir"       => get(ENV, "FORCING_DIR", joinpath(homedir(), "JRA55_data")),
                                   "filename_prefix"   => "sysimage_trace",
                                   "simulation_length" => trace_length,
                                   "sampling_length"   => trace_length,
                                   "use_gm"            => use_gm)

    @info "precompile_workload: tracing a 10-step forward model run (use_gm=$use_gm) in $(config_dict["output_dir"])"
    run_CATKE_GM_calibration_orca(Dict{String,Float64}(), Dict{String,Float64}(), config_dict)
    @info "precompile_workload: run! returned cleanly (use_gm=$use_gm)"
    return nothing
end

try
    # Trace both calibration configurations into the one sysimage.
    for use_gm in (true, false)
        _precompile_workload(use_gm)
    end
catch e
    @warn "precompile_workload failed; sysimage will still bake everything compiled so far" exception = (e, catch_backtrace())
end
