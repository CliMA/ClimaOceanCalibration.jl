using ClimaOceanCalibration.GMCalibration
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses: I, ParameterDistributions.constrained_gaussian
using TOML

output_dir = "./test_calibration_output"
mkpath(output_dir)
ensemble_size = 5

function CAL.forward_model(iteration, number)
    member_path = CAL.path_to_ensemble_member(output_dir, iteration, number)
    parameter_path = CAL.parameter_path(output_dir, iteration, number)

    config = TOML.parsefile(parameter_path)

    κ_skew = parse(Float64, config["κ_skew"])
    κ_symmetric = parse(Float64, config["κ_symmetric"])

    simulation = create_one_degree_omip(κ_skew = κ_skew, κ_symmetric = κ_symmetric, output_dir = member_path)
    run!(simulation)
    
    return simulation
end

function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        data = process_member_data(simdir_path)
        G[:, m] .= data
    end
    return G_ensemble
end

function process_member_data(simdir_path)

end