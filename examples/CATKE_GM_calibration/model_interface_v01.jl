# model_interface.jl
# ClimaCalibrate hook-up for the ORCA CATKE+GM calibration.
#
# Variables expected to be defined before this file is included:
#   - output_dir           (calibration top-level dir)
# The remaining metadata (output_dim, ensemble_size, lat_range, z_min,
# sampling_length, calibrated parameter names, woa_file, filename_prefix)
# are loaded from calibration_metadata.jld2 when not already defined by
# the controller, mirroring the pattern in
# examples/CATKE_calibration/model_interface.jl.

using ClimaCalibrate
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses: get_ϕ, get_ϕ_mean_final, get_error
using TOML
using JLD2
using Dates

include(joinpath(@__DIR__, "forward_model_orca.jl"))
include(joinpath(@__DIR__, "data_processing.jl"))
include(joinpath(@__DIR__, "calibration_plots.jl"))

if !@isdefined(output_dir)
    error("output_dir must be defined before including model_interface.jl")
end

metadata_file = joinpath(output_dir, "calibration_metadata.jld2")

macro _load_or_default(name)
    quote
        if !isdefined(@__MODULE__, $(QuoteNode(name)))
            $(esc(name)) = jldopen(metadata_file, "r") do file
                file[$(string(name))]
            end
            @info "Loaded $($(string(name)))=$($(esc(name))) from metadata"
        end
    end
end

@_load_or_default ensemble_size
@_load_or_default output_dim
@_load_or_default latitude_range
@_load_or_default z_min
@_load_or_default sampling_length
@_load_or_default simulation_length
@_load_or_default filename_prefix
if !isdefined(@__MODULE__, :staging_dir)
    staging_dir = jldopen(metadata_file, "r") do file
        haskey(file, "staging_dir") ? file["staging_dir"] : nothing
    end
    @info "Loaded staging_dir=$staging_dir from metadata"
end
@_load_or_default woa_file
@_load_or_default catke_param_names
@_load_or_default gm_param_names
@_load_or_default use_gm
@_load_or_default with_ice_dynamics

"""
    forward_model(iteration, member)

Read the per-member TOML parameter file, split keys into CATKE / GM
scalings, and dispatch to `run_CATKE_GM_calibration_orca`.
"""
function ClimaCalibrate.forward_model(iteration, member)
    member_path    = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    mkpath(member_path)

    parameter_path = ClimaCalibrate.parameter_path(output_dir, iteration, member)
    params = TOML.parsefile(parameter_path)

    catke_scalings = Dict{String,Float64}()
    gm_scalings    = Dict{String,Float64}()
    for name in catke_param_names
        catke_scalings[name] = params[name]["value"]
    end
    for name in gm_param_names
        gm_scalings[name] = params[name]["value"]
    end

    config_dict = Dict{String,Any}(
        "output_dir"        => member_path,
        "filename_prefix"   => filename_prefix,
        "iteration"         => iteration,
        "member"            => member,
        "simulation_length" => simulation_length,
        "sampling_length"   => sampling_length,
        "staging_dir"       => staging_dir,
        "use_gm"            => use_gm,
        "with_ice_dynamics" => with_ice_dynamics,
    )

    @info "iter=$iteration member=$member CATKE scalings=$catke_scalings GM scalings=$gm_scalings"

    try
        run_CATKE_GM_calibration_orca(catke_scalings, gm_scalings, config_dict)
    catch e
        error_file = joinpath(member_path, "RUN_FAILED.err")
        open(error_file, "w") do io
            println(io, "Run failed at $(now())")
            println(io, "CATKE scalings: $catke_scalings")
            println(io, "GM scalings:    $gm_scalings")
            println(io, "Error: $e")
            for (exc, bt) in Base.catch_stack()
                showerror(io, exc, bt); println(io)
            end
        end
        @error "Forward model failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
    return nothing
end

"""
    observation_map(iteration)

Build the G ensemble matrix (output_dim × ensemble_size). Failed runs
become NaN columns; ClimaCalibrate handles those downstream.
"""
function ClimaCalibrate.observation_map(iteration)
    G = zeros(output_dim, ensemble_size)
    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping failed member $m (iter $iteration)"
            G[:, m] .= NaN
        else
            try
                G[:, m] .= process_member_data(member_path, filename_prefix;
                                               lat_range = latitude_range, z_min)
            catch e
                @warn "process_member_data failed for member $m" exception=e
                G[:, m] .= NaN
            end
        end
    end
    return G
end

"""
    analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)

Save EKP diagnostics and render per-member figures vs WOA.
"""
function ClimaCalibrate.analyze_iteration(ekp, g_ensemble, prior, calib_output_dir, iteration)
    @info "Iteration $iteration completed"
    @info "Mean constrained parameters: $(get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error:    $(last(get_error(ekp)))"

    ϕs           = get_ϕ(prior, ekp)
    model_error  = get_error(ekp)
    param_names  = vcat(collect(catke_param_names), collect(gm_param_names))

    jldopen(joinpath(calib_output_dir, "ekp_diagnostics_iteration$(iteration).jld2"), "w") do file
        file["ϕs"]          = ϕs
        file["g_ensemble"]  = g_ensemble
        file["prior"]       = prior
        file["ekp"]         = ekp
        file["model_error"] = model_error
        file["param_names"] = param_names
    end

    ϕ = ϕs[iteration + 1]
    for m in 1:size(ϕ, 2)
        parts = ["$(name)=$(round(ϕ[i, m], digits=4))" for (i, name) in enumerate(param_names)]
        @info "iter=$iteration member $m: " * join(parts, ", ")
    end

    iter_fig_root = joinpath(ClimaCalibrate.path_to_iteration(calib_output_dir, iteration),
                             "figures")
    mkpath(iter_fig_root)
    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(calib_output_dir, iteration, m)
        # Skip members that produced no output to plot. Two cases:
        #  - the forward model caught an error and wrote RUN_FAILED.err, or
        #  - the whole batch job died outside Julia (node/GPU/sbatch failure),
        #    leaving the member dir empty with no marker at all.
        # Either way there are no *_means.jld2 / *_average.jld2 files, so every
        # sub-plot would warn; skip with one line instead.
        if isfile(joinpath(member_path, "RUN_FAILED.err")) || !member_has_output(member_path, filename_prefix)
            @info "Skipping figures for member $m (iter $iteration): no output (failed or never ran)"
            continue
        end
        fig_dir = joinpath(iter_fig_root, "member_$(m)")
        try
            plot_member_vs_woa(member_path, filename_prefix, woa_file,
                               latitude_range, z_min, sampling_length, fig_dir)
        catch e
            @warn "plot_member_vs_woa failed for member $m" exception=e
        end
    end
    return nothing
end
