using ClimaCalibrate
using ClimaOceanCalibration.DataWrangling
using Oceananigans
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using JLD2
using Glob
using Statistics
include("data_processing.jl")
include("gcloud_configuration.jl")

const output_dir = joinpath(pwd(), "calibration_runs", "test_run_gm")

function process_observation(obs_path, vertical_weighting=no_tapering)
    T_filepath = joinpath(obs_path, "T.jld2")
    S_filepath = joinpath(obs_path, "S.jld2")
    
    T_afts = jldopen(T_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    S_afts = jldopen(S_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    T_data = T_afts.data
    S_data = S_afts.data

    T_section = extract_southern_ocean_section(T_data, vertical_weighting)
    S_section = extract_southern_ocean_section(S_data, vertical_weighting)

    return vcat(T_section[.!isnan.(T_section)], S_section[.!isnan.(S_section)])
end

n_iterations = 3
κ_skew_prior = constrained_gaussian("κ_skew", 5e2, 3e2, 0, Inf)
κ_symmetric_prior = constrained_gaussian("κ_symmetric", 5e2, 3e2, 0, Inf)

priors = combine_distributions([κ_skew_prior, κ_symmetric_prior])

obs_paths = abspath.(glob("10yearaverage_2degree*", joinpath("calibration_data", "ECCO4Monthly")))
calibration_target_obs_path = obs_paths[findfirst(x -> occursin("2002", x), obs_paths)]

Y = hcat(process_observation.(obs_paths)...)

n_trials = 2
# the noise estimated from the samples (will have rank n_trials-1)
internal_cov = tsvd_cov_from_samples(Y) # SVD object

# the "5%" model error (diagonal)
model_error_frac = 0.05
data_mean = vec(mean(Y,dims=2))
model_error_cov = Diagonal((model_error_frac*data_mean).^2)

# regularize the model error diagonal (in case of zero entries)
model_error_cov += 1e-6*I

# Combine...
covariance = SVDplusD(internal_cov, model_error_cov)

Y_obs = Observation(Dict("samples" => process_observation(calibration_target_obs_path, taper_interior_ocean),
                         "covariances" => covariance,
                         "names" => basename(calibration_target_obs_path)))

utki = EnsembleKalmanProcess(Y_obs, TransformUnscented(priors))

backend = ClimaOceanSingleGPUGCPBackend

hpc_kwargs = hpc_kwargs = Dict(:ntasks => 1,
                               :cpus_per_task => 4,
                               :gpus_per_task => 1,
                               :mem => "128G",
                               :time => 120)

model_interface = abspath("./examples/GM_calibration/model_interface.jl")

ClimaCalibrate.calibrate(backend, utki, n_iterations, priors, output_dir; hpc_kwargs, verbose=true, model_interface)