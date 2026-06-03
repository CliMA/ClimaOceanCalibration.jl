# precompile_workload.jl
#
# PackageCompiler precompile-execution trace for the CATKE+GM ORCA calibration
# forward model. This script is passed to `create_sysimage(...;
# precompile_execution_file = ...)` by build_sysimage.jl. PackageCompiler runs
# it and records every method that gets compiled, baking that native code into
# the sysimage so each cold per-member calibration process starts hot.
#
# It mirrors the production build in forward_model_orca.jl
# (run_CATKE_GM_calibration_orca → omip_simulation(:orca; arch=GPU()) → run!)
# as closely as possible, using identical kwargs so the same GPU-typed method
# specializations are traced. Only the run length differs: we step the model a
# couple of times (stop_iteration = 2) instead of integrating for years.
#
# Requirements (true on a3mega H100 nodes):
#   - a functional GPU (the workload uses arch = GPU())
#   - the real JRA55 / WOA-ECCO / ORCA bathymetry input data, located via the
#     FORCING_DIR / RESTORING_DIR env vars (defaults match forward_model_orca.jl)
#
# A failure here is non-fatal to the sysimage build: build_sysimage.jl still
# bakes everything compiled up to the point of failure. We therefore wrap the
# body in try/catch and only log.

using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations
using Oceananigans
using Oceananigans.Units
using Dates

# Reuse build_catke_parameters / build_gm_parameters / attach_calibration_output_writers!
# and the same `using` set as the real forward model.
include(joinpath(@__DIR__, "forward_model_orca.jl"))

function _precompile_workload()
    forcing_dir   = get(ENV, "FORCING_DIR",   joinpath(homedir(), "JRA55_data"))
    restoring_dir = get(ENV, "RESTORING_DIR", joinpath(homedir(), "ECCO_data"))
    output_dir    = mktempdir(; prefix = "sysimage_trace_")

    # Default (scaling = 1) physics — exercises the same construction path as a
    # real member with no parameter overrides.
    catke_parameters = build_catke_parameters(Dict{String,Float64}())
    gm_parameters    = build_gm_parameters(Dict{String,Float64}())

    @info "precompile_workload: building omip_simulation(:orca; GPU()) for a 2-step trace"
    @info "  forcing_dir   = $forcing_dir"
    @info "  restoring_dir = $restoring_dir"
    @info "  output_dir    = $output_dir"

    sim = omip_simulation(:orca;
                          arch  = GPU(),
                          Nz    = 70,
                          depth = 5500,
                          catke_parameters,
                          gm_parameters,
                          biharmonic_timescale = 50days,
                          flux_configuration   = :corrected,
                          with_snow            = true,
                          with_ice_dynamics    = false,
                          diagnostics          = false,
                          Δt              = 30minutes,
                          forcing_dir,
                          restoring_dir,
                          staging_dir = nothing,
                          output_dir,
                          filename_prefix = "sysimage_trace")

    # Same output writers as the real run so their (de)serialization paths are
    # traced too. sampling_window is irrelevant for a 10-step run.
    attach_calibration_output_writers!(sim, output_dir, "sysimage_trace";
                                       stop_time       = 10 * 30minutes,
                                       sampling_window = 1 * 365days)

    # Step a couple of times to compile the time-stepping + closure kernels
    # (the expensive GPU compilation), then stop.
    sim.stop_iteration = 10
    sim.stop_time      = Inf
    run!(sim)

    @info "precompile_workload: run! returned cleanly after 2 steps"
    return nothing
end

try
    _precompile_workload()
catch e
    @warn "precompile_workload failed; sysimage will still bake everything compiled so far" exception = (e, catch_backtrace())
end
