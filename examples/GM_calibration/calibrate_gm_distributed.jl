const ensemble_size = 5
using Distributed

# Add workers with pre-set environment variables
nprocs = ensemble_size
addprocs(nprocs)
@everywhere @info "Worker $(myid())"
@everywhere ENV["CUDA_VISIBLE_DEVICES"] = myid() - 1

# Now load CUDA on all workers
@everywhere using CUDA
# Verify each worker sees exactly one GPU
@everywhere println("Worker $(myid()) sees GPU: $(CUDA.NVML.index(CUDA.NVML.Device(CUDA.uuid(CUDA.device()))))")

@everywhere begin
    using ClimaCalibrate
    using Distributed
    using ClimaOceanCalibration.DataWrangling
    using Oceananigans
    using EnsembleKalmanProcesses
    using EnsembleKalmanProcesses.ParameterDistributions
    using LinearAlgebra
    using JLD2
    using Glob
    using Statistics
    import ClimaCalibrate: generate_sbatch_script
    include(joinpath(pwd(), "examples", "GM_calibration", "data_processing.jl"))
    include(joinpath(pwd(), "examples", "GM_calibration", "model_interface.jl"))

    const output_dir = joinpath(pwd(), "calibration_runs", "gm_20year_ecco_distributed")
end

n_iterations = 5

κ_skew_prior = constrained_gaussian("κ_skew", 5e2, 3e2, 0, Inf)
κ_symmetric_prior = constrained_gaussian("κ_symmetric", 5e2, 3e2, 0, Inf)

priors = combine_distributions([κ_skew_prior, κ_symmetric_prior])

obs_paths = abspath.(vcat(glob("10yearaverage_2degree*", joinpath("calibration_data", "ECCO4Monthly")),
                          glob("10yearaverage_2degree*", joinpath("calibration_data", "EN4Monthly"))))

calibration_target_obs_path = abspath(joinpath("calibration_data", "ECCO4Monthly", "10yearaverage_2degree2002-01-01T00-00-00"))

# synthetic_obs_paths = abspath.(glob("*500.0_500.0*20year*", joinpath("calibration_data", "synthetic_observations")))
# Y = hcat(process_observation.(obs_paths, no_tapering)..., process_member_data.(synthetic_obs_paths, no_tapering)...)
Y = hcat(process_observation.(obs_paths, no_tapering)...)

const output_dim = size(Y, 1)

n_trials = size(Y, 2)

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

backend = ClimaCalibrate.WorkerBackend

simulation_length = 20
sampling_length = 10
ClimaCalibrate.forward_model(iteration, member) = ClimaCalibrate.forward_model(iteration, member; simulation_length, sampling_length)

ClimaCalibrate.calibrate(ClimaCalibrate.WorkerBackend, utki, n_iterations, priors, output_dir)