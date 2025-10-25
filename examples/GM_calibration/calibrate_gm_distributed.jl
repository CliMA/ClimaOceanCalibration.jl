const ensemble_size = 5
using Distributed
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--simulation_length"
            help = "Length of calibration simulation in years"
            arg_type = Int
            default = 6
        "--sampling_length"
            help = "Length of sampling period in years"
            arg_type = Int
            default = 1
        "--zonal_average"
            help = "Whether to perform zonal averaging in loss function"
            arg_type = Bool
            default = false
        "--observation_covariance"
            help = "Type of covariance to use (observations vs predetermined)"
            arg_type = Bool
            default = true
        "--pickup"
            help = "Pickup files for simulation spinup"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

args = parse_commandline()

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
    using ClimaOcean
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

    args = $args

    const simulation_length = args["simulation_length"]
    const sampling_length = args["sampling_length"]
    const zonal_average = args["zonal_average"]
    const observation_covariance = args["observation_covariance"]
    const pickup = args["pickup"] ? nothing : Dict("ocean" => joinpath(pwd(), "pickups", "ocean_pickup.jld2"),
                                                   "sea_ice" => joinpath(pwd(), "pickups", "seaice_pickup.jld2"))

    obl_closure = ClimaOcean.OceanSimulations.default_ocean_closure()

    if obl_closure isa RiBasedVerticalDiffusivity
        obl_str = "RiBased"
    else
        obl_str = "CATKE"
    end

    if observation_covariance
        cov_str = "obscov"
    else
        cov_str = "diagcov"
    end

    const output_dir = joinpath(pwd(), "calibration_runs", "gm_$(simulation_length)yr_$(sampling_length)yravg_ecco_$(obl_str)_$(cov_str)$(zonal_average ? "_zonalavg" : "")")
    ClimaCalibrate.forward_model(iteration, member) = gm_forward_model(iteration, member; simulation_length, sampling_length, obl_closure, pickup)
    ClimaCalibrate.observation_map(iteration) = gm_construct_g_ensemble(iteration, zonal_average)
end

n_iterations = 10

κ_skew_prior = constrained_gaussian("κ_skew", 5e2, 3e2, 0, Inf)
κ_symmetric_prior = constrained_gaussian("κ_symmetric", 5e2, 3e2, 0, Inf)

priors = combine_distributions([κ_skew_prior, κ_symmetric_prior])

obs_paths = abspath.(glob("$(sampling_length)yearaverage_2degree*", joinpath("calibration_data", "ECCO4Monthly")))

calibration_target_obs_path = abspath(joinpath("calibration_data", "ECCO4Monthly", "$(sampling_length)yearaverage_2degree2007-01-01T00-00-00"))

Y = hcat(process_observation.(obs_paths, no_tapering, zonal_average)...)

const output_dim = size(Y, 1)

n_trials = size(Y, 2)

if observation_covariance
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
else
    T_variance = 0.7^2
    S_variance = 0.1^2
    N_data = output_dim ÷ 2
    covariance = Diagonal(vcat(fill(T_variance, N_data), fill(S_variance, N_data)))
end

Y_obs = Observation(Dict("samples" => process_observation(calibration_target_obs_path, taper_interior_ocean, zonal_average),
                         "covariances" => covariance,
                         "names" => basename(calibration_target_obs_path)))

utki = EnsembleKalmanProcess(Y_obs, TransformUnscented(priors))

backend = ClimaCalibrate.WorkerBackend

ClimaCalibrate.calibrate(ClimaCalibrate.WorkerBackend, utki, n_iterations, priors, output_dir)