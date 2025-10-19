using ClimaCalibrate
using TOML
using ClimaOceanCalibration.DataWrangling
using EnsembleKalmanProcesses
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using JLD2
include("half_degree_omip.jl")
include("data_processing.jl")
include("data_plotting.jl")

function ClimaCalibrate.forward_model(iteration, member; simulation_length, sampling_length)
    config_dict = Dict()
    
    # Set the output path for the current member
    member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path

    # Set the parameters for the current member
    parameter_path = ClimaCalibrate.parameter_path(output_dir, iteration, member)
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    config_dict["iteration"] = iteration
    config_dict["member"] = member
    config_dict["simulation_length"] = simulation_length
    config_dict["sampling_length"] = sampling_length

    params = TOML.parsefile(parameter_path)
    κ_skew = params["κ_skew"]
    κ_symmetric = params["κ_symmetric"]

    try
        # run_gm_calibration_omip_dry_run(κ_skew["value"], κ_symmetric["value"], config_dict)
        run_gm_calibration_omip(κ_skew["value"], κ_symmetric["value"], config_dict)
    catch e
        # Create a failure indicator file with error information
        error_file = joinpath(member_path, "RUN_FAILED.err")
        open(error_file, "w") do io
            println(io, "Run failed at $(now())")
            println(io, "Parameters: κ_skew = $(κ_skew), κ_symmetric = $(κ_symmetric)")
            println(io, "Error: $(e)")
            println(io, "Backtrace:")
            for (exc, bt) in Base.catch_stack()
                showerror(io, exc, bt)
                println(io)
            end
        end
        
        @error "GM calibration failed (κ_skew = $(κ_skew), κ_symmetric = $(κ_symmetric))" exception=(e, catch_backtrace())
    end

    return nothing
end

function ClimaCalibrate.observation_map(iteration)
    G_ensemble = zeros(output_dim, ensemble_size)

    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)

        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping member $m for iteration $iteration due to failed run."
            G_ensemble[:, m] .= NaN
        else
            G_ensemble[:, m] .= process_member_data(member_path, taper_interior_ocean)
        end
    end

    return G_ensemble
end

function ClimaCalibrate.analyze_iteration(ekp, g_ensemble, prior, output_dir, iteration)
    @info "Mean constrained parameter(s): $(get_ϕ_mean_final(prior, ekp))"
    @info "Covariance-weighted error: $(last(get_error(ekp)))"

    ϕs = get_ϕ_final(prior, ekp)

    compute_error!(ekp)
    avg_rmse = compute_average_rmse(ekp)
    error_metrics = get_error_metrics(ekp)

    jldopen(joinpath(output_dir, "ekp_diagnostics_iteration$(iteration).jld2"), "w") do file
        file["ϕs"] = ϕs
        file["avg_rmse"] = avg_rmse
        file["error_metrics"] = error_metrics
        file["g_ensemble"] = g_ensemble
        file["prior"] = prior
        file["ekp"] = ekp
        file["ϕ_mean"] = get_ϕ_mean_final(prior, ekp)
    end

    plots_filepath = abspath(joinpath(output_dir, "diagnostics_output"))
    mkpath(plots_filepath)

    fig = plot_parameter_distribution(ϕs, avg_rmse)
    save(joinpath(plots_filepath, "iteration_$(iteration)_parameter_distribution.png"), fig)

    obs_path = joinpath(pwd(), "calibration_data", "ECCO4Monthly", "10yearaverage_2degree2002-01-01T00-00-00")

    T_truth_filepath = joinpath(obs_path, "T.jld2")
    S_truth_filepath = joinpath(obs_path, "S.jld2")
    b_truth_filepath = joinpath(obs_path, "b.jld2")
    
    T_truth_afts = jldopen(T_truth_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    S_truth_afts = jldopen(S_truth_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    b_truth_afts = jldopen(b_truth_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    T_truth = on_architecture(CPU(), T_truth_afts.data)
    S_truth = on_architecture(CPU(), S_truth_afts.data)
    b_truth = on_architecture(CPU(), b_truth_afts.data)

    Nt_truth = length(T_truth.times)

    for i in 1:Nt_truth
        mask_immersed_field!(T_truth[i], NaN)
        mask_immersed_field!(S_truth[i], NaN)
        mask_immersed_field!(b_truth[i], NaN)
    end

    target_grid, regridder = jldopen(joinpath(pwd(), "examples", "GM_calibration", "grids_and_regridder.jld2"), "r") do file
        return file["target_grid"], file["regridder"]
    end

    for m in 1:ensemble_size
        @info "Plotting zonal averages for member $m"

        κ_skew, κ_symmetric = ϕs[:, m]
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)
        model_filepath = joinpath(member_path, "ocean_complete_fields_10year_average_calibrationsample.jld2")

        T_model = FieldTimeSeries(model_filepath, "T", backend=InMemory())
        S_model = FieldTimeSeries(model_filepath, "S", backend=InMemory())
        b_model = FieldTimeSeries(model_filepath, "b", backend=InMemory())

        T_model_field = CenterField(target_grid)
        S_model_field = CenterField(target_grid)
        b_model_field = CenterField(target_grid)

        Nt_model = length(T_model.times)

        mask_immersed_field!(T_model[Nt_model], NaN)
        mask_immersed_field!(S_model[Nt_model], NaN)
        mask_immersed_field!(b_model[Nt_model], NaN)

        regrid!(T_model_field, regridder, T_model[Nt_model])
        regrid!(S_model_field, regridder, S_model[Nt_model])
        regrid!(b_model_field, regridder, b_model[Nt_model])

        mask_immersed_field!(T_model_field, NaN)
        mask_immersed_field!(S_model_field, NaN)
        mask_immersed_field!(b_model_field, NaN)

        T_fig = plot_zonal_average(T_truth[Nt_truth], T_model_field, "T", κ_skew, κ_symmetric)
        S_fig = plot_zonal_average(S_truth[Nt_truth], S_model_field, "S", κ_skew, κ_symmetric)
        b_fig = plot_zonal_average(b_truth[Nt_truth], b_model_field, "b", κ_skew, κ_symmetric)

        save(joinpath(plots_filepath, "iter$(iteration)_member$(m)_skew_$(κ_skew)_sym_$(κ_symmetric)_T_zonal_average.png"), T_fig)
        save(joinpath(plots_filepath, "iter$(iteration)_member$(m)_skew_$(κ_skew)_sym_$(κ_symmetric)_S_zonal_average.png"), S_fig)
        save(joinpath(plots_filepath, "iter$(iteration)_member$(m)_skew_$(κ_skew)_sym_$(κ_symmetric)_b_zonal_average.png"), b_fig)
    end

end