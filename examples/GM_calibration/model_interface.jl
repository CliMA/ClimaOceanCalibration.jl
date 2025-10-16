using ClimaCalibrate
using TOML
using ClimaOceanCalibration.DataWrangling
include("half_degree_omip.jl")
include("data_processing.jl")

const output_dir = joinpath(pwd(), "calibration_runs", "test_run_gm")
const ensemble_size = 5
const output_dim = 314594

function ClimaCalibrate.forward_model(iteration, member)
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

    params = TOML.parsefile(parameter_path)
    κ_skew = params["κ_skew"]
    κ_symmetric = params["κ_symmetric"]

    try
        run_gm_calibration_omip_dry_run(κ_skew["value"], κ_symmetric["value"], config_dict)
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

    return simulation
end

function ClimaCalibrate.observation_map(iteration)
    G_ensemble = zeros(output_dim, ensemble_size)

    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)

        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping member $m for iteration $iteration due to failed run."
            G_ensemble[:, m] .= NaN
        else
            G_ensemble[:, m] .= process_member_data(member_path)
        end
    end

    return G_ensemble
end