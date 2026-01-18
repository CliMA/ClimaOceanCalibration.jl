# model_interface_module.jl
# Model interface for CATKE calibration using the CATKECalibration module
#
# This file defines the forward_model and observation_map functions required by
# ClimaCalibrate. It uses the CATKECalibration module which provides precompiled
# functions for running the forward model and processing data.
#
# The main calibration script (calibrate_catke_module.jl) defines output_dir,
# ensemble_size, output_dim, and zonal_average before including this file.
using Pkg
Pkg.activate("CATKECalibration")

using CATKECalibration
using ClimaCalibrate
using TOML
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses: get_ϕ, get_ϕ_mean_final, get_error
using JLD2
using Dates

metadata_file = joinpath(pwd(), "examples", "CATKE_calibration", "calibration_metadata.jld2")

if !@isdefined(output_dir)
    @info "Loading output_dir from metadata file..."
    output_dir = jldopen(metadata_file, "r") do file
        return file["output_dir"]
    end
    @info "Loaded: output_dir=$output_dir"
else
    @info "Using output_dir from parent scope: $output_dir"
end

if !@isdefined(zonal_average)
    @info "Loading zonal_average from metadata file..."
    zonal_average = jldopen(metadata_file, "r") do file
        return file["zonal_average"]
    end
    @info "Loaded: zonal_average=$zonal_average"
else
    @info "Using zonal_average from parent scope: $zonal_average"
end

if !@isdefined(ensemble_size)
    @info "Loading ensemble_size from metadata file..."
    ensemble_size = jldopen(metadata_file, "r") do file
        return file["ensemble_size"]
    end
    @info "Loaded: ensemble_size=$ensemble_size"
else
    @info "Using ensemble_size from parent scope: $ensemble_size"
end

if !@isdefined(output_dim)
    @info "Loading output_dim from metadata file..."
    output_dim = jldopen(metadata_file, "r") do file
        return file["output_dim"]
    end
    @info "Loaded: output_dim=$output_dim"
else
    @info "Using output_dim from parent scope: $output_dim"
end

"""
    ClimaCalibrate.forward_model(iteration, member)

Run a single forward model for CATKE calibration.

Reads 5 CATKE scaling parameters from the parameter TOML file:
- Cˢ_scaling: Surface layer TKE production scaling
- Cᵘⁿ_scaling: Unstable convective mixing scaling
- Cᶜ_scaling: Stable/convective mixing scaling
- Cˢᵖ_scaling: Shear production scaling
- Cᵉc_scaling: TKE equation scaling

Uses the precompiled run_CATKE_calibration_omip function from CATKECalibration module.
"""
function ClimaCalibrate.forward_model(iteration, member)
    config_dict = Dict{String, Any}()

    # Set the output path for the current member
    member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    mkpath(member_path)
    config_dict["output_dir"] = member_path

    # Set the parameters for the current member
    parameter_path = ClimaCalibrate.parameter_path(output_dir, iteration, member)
    config_dict["toml"] = [parameter_path]

    config_dict["iteration"] = iteration
    config_dict["member"] = member

    # Read CATKE scaling parameters from TOML
    params = TOML.parsefile(parameter_path)

    Cˢ_scaling = params["Cˢ_scaling"]["value"]
    Cᵘⁿ_scaling = params["Cᵘⁿ_scaling"]["value"]
    Cᶜ_scaling = params["Cᶜ_scaling"]["value"]
    Cˢᵖ_scaling = params["Cˢᵖ_scaling"]["value"]
    Cᵉc_scaling = params["Cᵉc_scaling"]["value"]

    @info "Running CATKE forward model: iteration=$iteration, member=$member"
    @info "Parameters: Cˢ=$Cˢ_scaling, Cᵘⁿ=$Cᵘⁿ_scaling, Cᶜ=$Cᶜ_scaling, Cˢᵖ=$Cˢᵖ_scaling, Cᵉc=$Cᵉc_scaling"

    try
        # Use the precompiled forward model from CATKECalibration module
        run_CATKE_calibration_omip(Cˢ_scaling, Cᵘⁿ_scaling, Cᶜ_scaling, Cˢᵖ_scaling, Cᵉc_scaling, config_dict)
    catch e
        # Create a failure indicator file with error information
        error_file = joinpath(member_path, "RUN_FAILED.err")
        open(error_file, "w") do io
            println(io, "Run failed at $(now())")
            println(io, "Parameters: Cˢ_scaling=$Cˢ_scaling, Cᵘⁿ_scaling=$Cᵘⁿ_scaling, Cᶜ_scaling=$Cᶜ_scaling, Cˢᵖ_scaling=$Cˢᵖ_scaling, Cᵉc_scaling=$Cᵉc_scaling")
            println(io, "Error: $(e)")
            println(io, "Backtrace:")
            for (exc, bt) in Base.catch_stack()
                showerror(io, exc, bt)
                println(io)
            end
        end

        @error "CATKE calibration failed" exception=(e, catch_backtrace())
        rethrow(e)
    end

    return nothing
end

"""
    ClimaCalibrate.observation_map(iteration)

Construct the G ensemble matrix from forward model outputs.

Returns a matrix of size (output_dim × ensemble_size) where each column
contains the processed model output for one ensemble member.

Note: dz weighting is applied to match the weighting applied to Y_target.
Uses the process_member_data function from CATKECalibration module.
"""
function ClimaCalibrate.observation_map(iteration)
    G_ensemble = zeros(output_dim, ensemble_size)

    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)

        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping member $m for iteration $iteration due to failed run."
            G_ensemble[:, m] .= NaN
        else
            try
                # Use the process_member_data function from CATKECalibration module
                G_ensemble[:, m] .= process_member_data(member_path, zonal_average; apply_dz_weighting=true)
            catch e
                @warn "Failed to process member $m for iteration $iteration: $e"
                G_ensemble[:, m] .= NaN
            end
        end
    end

    return G_ensemble
end

"""
    ClimaCalibrate.analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)

Custom analysis callback run after each EKP iteration.
Saves diagnostics and parameter distributions.
"""
function ClimaCalibrate.analyze_iteration(ekp, g_ensemble, prior, calib_output_dir, iteration)
    @info "Iteration $iteration completed"
    @info "Mean constrained parameter(s): $(get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error: $(last(get_error(ekp)))"

    ϕs = get_ϕ(prior, ekp)
    model_error = get_error(ekp)

    # Save diagnostics
    jldopen(joinpath(calib_output_dir, "ekp_diagnostics_iteration$(iteration).jld2"), "w") do file
        file["ϕs"] = ϕs
        file["g_ensemble"] = g_ensemble
        file["prior"] = prior
        file["ekp"] = ekp
    end

    # Create diagnostics output directory
    plots_filepath = abspath(joinpath(calib_output_dir, "diagnostics_output"))
    mkpath(plots_filepath)

    # Log parameter values for each member
    ϕ = ϕs[iteration + 1]
    param_names = ["Cˢ_scaling", "Cᵘⁿ_scaling", "Cᶜ_scaling", "Cˢᵖ_scaling", "Cᵉc_scaling"]

    for m in 1:size(ϕ, 2)
        params_str = join(["$(name)=$(round(ϕ[i, m], digits=4))" for (i, name) in enumerate(param_names)], ", ")
        @info "Member $m: $params_str"
    end
end
