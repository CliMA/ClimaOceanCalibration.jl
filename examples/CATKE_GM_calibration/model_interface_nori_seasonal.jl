# model_interface_nori_seasonal.jl
# ClimaCalibrate hook-up for the SEASONAL-CYCLE ORCA NORi calibration.
# Mirrors model_interface_seasonal.jl but drives NORi rather than CATKE:
#   - uses `run_NORi_calibration_orca` in the forward model, and
#   - reads `nori_param_names` from calibration metadata.
#
# `output_dir` must be defined before this file is included.

using ClimaCalibrate
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses: get_ϕ, get_ϕ_mean_final, get_error
using TOML
using JLD2
using Dates

include(joinpath(@__DIR__, "forward_model_orca.jl"))
isdefined(@__MODULE__, :process_member_data_seasonal) ||
    include(joinpath(@__DIR__, "data_processing_seasonal.jl"))
include(joinpath(@__DIR__, "seasonal_plots.jl"))
isdefined(@__MODULE__, :plot_member_vs_woa) ||
    include(joinpath(@__DIR__, "calibration_plots.jl"))

if !@isdefined(output_dir)
    error("output_dir must be defined before including model_interface_nori_seasonal.jl")
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
@_load_or_default nori_param_names
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

struct NORiSeasonalInterface <: ClimaCalibrate.AbstractModelInterface
    model_interface_path::String
    project_path::String
end

function NORiSeasonalInterface(;
    model_interface_path = abspath(@__FILE__),
    project_path         = abspath(joinpath(@__DIR__, "..", "..")),
)
    return NORiSeasonalInterface(model_interface_path, project_path)
end

ClimaCalibrate.model_interface_filepath(i::NORiSeasonalInterface) = i.model_interface_path
ClimaCalibrate.experiment_dir(i::NORiSeasonalInterface)            = i.project_path

function ClimaCalibrate.forward_model(::NORiSeasonalInterface, iteration, member)
    member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    mkpath(member_path)

    parameter_path = ClimaCalibrate.parameter_path(output_dir, iteration, member)
    params = TOML.parsefile(parameter_path)

    nori_scalings = Dict{String,Float64}()
    gm_scalings   = Dict{String,Float64}()
    for name in nori_param_names
        nori_scalings[name] = params[name]["value"]
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
        "output_mode"       => "seasonal",
    )

    @info "iter=$iteration member=$member (NORi seasonal) NORi=$nori_scalings GM=$gm_scalings"

    try
        run_NORi_calibration_orca(nori_scalings, gm_scalings, config_dict)
    catch e
        error_file = joinpath(member_path, "RUN_FAILED.err")
        open(error_file, "w") do io
            println(io, "Run failed at $(now())")
            println(io, "NORi scalings: $nori_scalings")
            println(io, "GM scalings:   $gm_scalings")
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

function ClimaCalibrate.observation_map(::NORiSeasonalInterface, iteration)
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

function ClimaCalibrate.analyze_iteration(::NORiSeasonalInterface, ekp, g_ensemble, prior, calib_output_dir, iteration)
    @info "Iteration $iteration completed (NORi seasonal)"
    @info "Mean constrained parameters: $(get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error:    $(last(get_error(ekp)))"

    ϕs          = get_ϕ(prior, ekp)
    model_error = get_error(ekp)
    param_names = vcat(collect(nori_param_names), collect(gm_param_names))

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

    iter_fig_root = joinpath(ClimaCalibrate.path_to_iteration(calib_output_dir, iteration), "figures")
    mkpath(iter_fig_root)

    annual_woa_file = abspath(joinpath(@__DIR__, "calibration_data",
        Δz_top === nothing ? "woa_orca_grid.jld2" : "woa_orca_grid_dztop$(Δz_top).jld2"))
    has_annual_woa = isfile(annual_woa_file)
    has_annual_woa || @warn "Annual WOA-on-ORCA cache not found; skipping annual-style \
        per-member figures (run precompute_woa_orca.jl to enable them)" annual_woa_file

    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(calib_output_dir, iteration, m)
        if isfile(joinpath(member_path, "RUN_FAILED.err")) ||
           !member_has_seasonal_output(member_path, filename_prefix)
            @info "Skipping seasonal video for member $m (iter $iteration): no output"
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
        if has_annual_woa
            try
                plot_member_vs_woa(member_path, filename_prefix, annual_woa_file,
                                   latitude_range, z_min, sampling_length, fig_dir)
            catch e
                @warn "plot_member_vs_woa failed for member $m" exception=e
            end
        end
    end
    return nothing
end
