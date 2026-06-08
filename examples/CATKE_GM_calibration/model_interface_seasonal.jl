# model_interface_seasonal.jl
# ClimaCalibrate hook-up for the SEASONAL-CYCLE ORCA CATKE+GM calibration.
# Mirrors model_interface.jl but:
#   - the forward model runs with output_mode=:seasonal (so it writes the monthly
#     3-D T,S,b and surface E/P files; init stays WOA Annual), and
#   - the observation map builds the zonal-mean seasonal-cycle vector (last 12
#     months, regridded to 1° lat-lon, tropics/upper-200 m) via
#     data_processing_seasonal.jl.
#
# `output_dir` must be defined before this file is included (same contract as
# model_interface.jl).

using ClimaCalibrate
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses: get_ϕ, get_ϕ_mean_final, get_error
using TOML
using JLD2
using Dates

include(joinpath(@__DIR__, "forward_model_orca.jl"))
# Guard against double-inclusion of the seasonal data file (the parent calibrate
# script includes it early; non-isbits consts in it must not be redefined).
isdefined(@__MODULE__, :process_member_data_seasonal) ||
    include(joinpath(@__DIR__, "data_processing_seasonal.jl"))
include(joinpath(@__DIR__, "seasonal_plots.jl"))

if !@isdefined(output_dir)
    error("output_dir must be defined before including model_interface_seasonal.jl")
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
if !isdefined(@__MODULE__, :Δz_top)
    Δz_top = jldopen(metadata_file, "r") do file
        haskey(file, "Δz_top") ? file["Δz_top"] : nothing
    end
    @info "Loaded Δz_top=$Δz_top from metadata"
end
if !isdefined(@__MODULE__, :skin_temperature)
    skin_temperature = jldopen(metadata_file, "r") do file
        haskey(file, "skin_temperature") ? file["skin_temperature"] : false
    end
    @info "Loaded skin_temperature=$skin_temperature from metadata"
end

struct CATKEGMSeasonalInterface <: ClimaCalibrate.AbstractModelInterface
    model_interface_path::String
    project_path::String
end

function CATKEGMSeasonalInterface(;
    model_interface_path = abspath(@__FILE__),
    project_path         = abspath(joinpath(@__DIR__, "..", "..")),
)
    return CATKEGMSeasonalInterface(model_interface_path, project_path)
end

ClimaCalibrate.model_interface_filepath(i::CATKEGMSeasonalInterface) = i.model_interface_path
ClimaCalibrate.experiment_dir(i::CATKEGMSeasonalInterface)            = i.project_path

"""
    forward_model(::CATKEGMSeasonalInterface, iteration, member)

Read the per-member TOML parameters and run the ORCA forward model in seasonal
mode (monthly init + monthly 3-D T,S output).
"""
function ClimaCalibrate.forward_model(::CATKEGMSeasonalInterface, iteration, member)
    member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
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
        "Δz_top"            => Δz_top,
        "skin_temperature"  => skin_temperature,
        # Seasonal-cycle specific (model still initializes from WOA Annual).
        "output_mode"       => "seasonal",
    )

    @info "iter=$iteration member=$member (seasonal) CATKE=$catke_scalings GM=$gm_scalings"

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
    observation_map(::CATKEGMSeasonalInterface, iteration)

Build the G ensemble matrix (output_dim × ensemble_size) from the zonal-mean
seasonal cycle of each member. Failed runs become NaN columns.
"""
function ClimaCalibrate.observation_map(::CATKEGMSeasonalInterface, iteration)
    G = zeros(output_dim, ensemble_size)
    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping failed member $m (iter $iteration)"
            G[:, m] .= NaN
        else
            try
                G[:, m] .= process_member_data_seasonal(member_path, filename_prefix;
                                                        lat_range = latitude_range, z_min)
            catch e
                @warn "process_member_data_seasonal failed for member $m" exception=e
                G[:, m] .= NaN
            end
        end
    end
    return G
end

"""
    analyze_iteration(::CATKEGMSeasonalInterface, ekp, g_ensemble, prior, output_dir, iteration)

Save EKP diagnostics, log the constrained parameters, and render a per-member
seasonal-cycle video (zonal-mean T/S: WOA | sim | difference, last-year cycle).
"""
function ClimaCalibrate.analyze_iteration(::CATKEGMSeasonalInterface, ekp, g_ensemble, prior, calib_output_dir, iteration)
    @info "Iteration $iteration completed (seasonal)"
    @info "Mean constrained parameters: $(get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error:    $(last(get_error(ekp)))"

    ϕs          = get_ϕ(prior, ekp)
    model_error = get_error(ekp)
    param_names = vcat(collect(catke_param_names), collect(gm_param_names))

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

    # Per-member seasonal-cycle videos (zonal-mean T/S vs WOA monthly).
    iter_fig_root = joinpath(ClimaCalibrate.path_to_iteration(calib_output_dir, iteration), "figures")
    mkpath(iter_fig_root)
    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(calib_output_dir, iteration, m)
        if isfile(joinpath(member_path, "RUN_FAILED.err")) ||
           !member_has_seasonal_output(member_path, filename_prefix)
            @info "Skipping seasonal video for member $m (iter $iteration): no output (failed or never ran)"
            continue
        end
        fig_dir = joinpath(iter_fig_root, "member_$(m)")
        mkpath(fig_dir)
        try
            plot_member_seasonal_video(member_path, filename_prefix, woa_file,
                                       latitude_range, z_min,
                                       joinpath(fig_dir, "seasonal_zonal_TSb.mp4"))
        catch e
            @warn "plot_member_seasonal_video failed for member $m" exception=e
        end
        try
            plot_member_seasonal_EP_video(member_path, filename_prefix,
                                          latitude_range,
                                          joinpath(fig_dir, "seasonal_EP_maps.mp4"))
        catch e
            @warn "plot_member_seasonal_EP_video failed for member $m" exception=e
        end
    end
    return nothing
end
