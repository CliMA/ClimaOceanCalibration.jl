# calibrate_catke.jl
# Main calibration script for CATKE parameter estimation using BatchedSlurmGCPBackend
#
# This script calibrates 5 CATKE scaling parameters using Ensemble Kalman Inversion (EKI)
# against monthly-averaged ECCO4 temperature and salinity observations.
#
# The calibration uses the batched Slurm backend, which submits jobs that run multiple
# ensemble members in parallel on a single GCP a3mega node (8 GPUs per node).
#
# Usage:
#   julia --project calibrate_catke.jl
#
# The script will:
# 1. Define priors for 5 CATKE scaling parameters
# 2. Load 1993 monthly-averaged ECCO4 observations as calibration target
# 3. Build observation covariance from all available years (1992-2017)
# 4. Run EKI calibration with batched Slurm backend

using ClimaCalibrate
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using JLD2
using Glob
using Statistics
# using CairoMakie
# import EnsembleKalmanProcesses.Visualize as viz

# Include the batched Slurm backend
include(joinpath(@__DIR__, "batched_slurm_backend.jl"))

# Include data processing utilities
include(joinpath(@__DIR__, "data_processing.jl"))

# ============================================
# Calibration Configuration
# ============================================

# Ensemble configuration
# With 5 parameters and TransformUnscented, we get 2*5+1 = 11 ensemble members
const n_iterations = 10
const prior_std = 1
const T_std = 0.5
const S_std = T_std / 4

# Data processing options
const zonal_average = false  # Use full 3D fields, not zonal averages
const latitude_range = [(-52, -20), (20, 52)]  # Extratropical regions (Southern: -52 to -20, Northern: 20 to 52)
const z_min = -500  # Upper 1000m only

# Output directory
output_dir = joinpath(pwd(), "calibration_runs", "catke_extratropics_$(latitude_range[end][1])_$(latitude_range[end][2])_zmin_$(z_min)_Tstd_$(T_std)_Sstd_$(S_std)_priorstd_$(prior_std)_symmetric")
mkpath(output_dir)

# ============================================
# Prior Definitions for CATKE Scaling Parameters
# ============================================
# These are scaling factors applied to the default CATKE parameter values.
# A scaling of 1.0 means using the default value.
# Priors are log-normal to ensure positive values.

Cˢ_prior = constrained_gaussian("Cˢ_scaling", 1.0, prior_std, 0, Inf)     # Surface layer TKE production
Cᵘⁿ_prior = constrained_gaussian("Cᵘⁿ_scaling", 1.0, prior_std, 0, Inf)   # Unstable/convective mixing
Cᶜ_prior = constrained_gaussian("Cᶜ_scaling", 1.0, prior_std, 0, Inf)     # Stable/convective mixing
Cˢᵖ_prior = constrained_gaussian("Cˢᵖ_scaling", 1.0, prior_std, 0, Inf)   # Shear production
Cᵉc_prior = constrained_gaussian("Cᵉc_scaling", 1.0, prior_std, 0, Inf)   # TKE equation

priors = combine_distributions([Cˢ_prior, Cᵘⁿ_prior, Cᶜ_prior, Cˢᵖ_prior, Cᵉc_prior])

#%%
# fig_priors = Figure(size = (1200, 600))
# viz.plot_parameter_distribution(fig_priors[1, 1], priors)

# fig_priors
#%%
# ============================================
# Load Observations
# ============================================
@info "Loading observations..."

# Path to monthly-averaged ECCO4 data
calibration_data_dir = joinpath(pwd(), "calibration_data", "ECCO4Monthly")

# Find all available observation years
obs_paths = abspath.(glob("monthlyaverage_4degree*", calibration_data_dir))

if isempty(obs_paths)
    error("""
        No observation data found in $calibration_data_dir.
        Please run average_ECCO_data.jl first to generate the monthly-averaged ECCO data.
        Expected directory pattern: monthlyaverage_4degree*
    """)
end

@info "Found $(length(obs_paths)) observation years"

# Use 1993 as calibration target (second year of ECCO data)
calibration_target_obs_path = filter(p -> occursin("1993", p), obs_paths)
if isempty(calibration_target_obs_path)
    error("Could not find 1993 observation data. Available: $(basename.(obs_paths))")
end
calibration_target_obs_path = first(calibration_target_obs_path)

@info "Calibration target: $(basename(calibration_target_obs_path))"

# Get the target observation (all 12 months of 1993 concatenated) without dz weighting
Y_target = vec(process_monthly_observations(calibration_target_obs_path, zonal_average; apply_dz_weighting=false, latitude_range, z_min))

# ============================================
# Build Observation Covariance
# ============================================
@info "Building diagonal observation covariance..."

# Build diagonal covariance with specified T and S standard deviations
covariance = build_diagonal_covariance(Y_target; T_variance=T_std^2, S_variance=S_std^2)

# Store output dimension for model interface
const output_dim = length(Y_target)
@info "Output dimension: $output_dim"

# Create Observation object
Y_obs = Observation(Dict(
    "samples" => Y_target,
    "covariances" => covariance,
    "names" => "ECCO4_monthly_1993"
))

# ============================================
# Create EKP
# ============================================
scheduler = DataMisfitController(on_terminate="continue")

@info "Creating EnsembleKalmanProcess..."
# ekp = EnsembleKalmanProcess(Y_obs, TransformUnscented(priors, sigma_points="simplex"); scheduler)
ekp = EnsembleKalmanProcess(Y_obs, TransformUnscented(priors); scheduler)

# Display initial ensemble parameters
function display_initial_parameters(priors, ekp)
    ϕ_all = get_ϕ(priors, ekp)
    ϕ = ϕ_all[end]  # get_ϕ returns a vector of matrices, one per iteration
    param_names = get_name(priors)
    n_params, n_ensemble = size(ϕ)

    @info "Initial ensemble parameters (iteration 0):"
    for i in 1:n_params
        vals = ϕ[i, :]
        @info "  $(param_names[i]): $(round.(vals, digits=4))"
    end
end

display_initial_parameters(priors, ekp)

const ensemble_size = EnsembleKalmanProcesses.get_N_ens(ekp)
n_batches = ceil(Int, ensemble_size / 8)

jldopen(joinpath(output_dir, "calibration_metadata.jld2"), "w") do file
    file["zonal_average"] = zonal_average
    file["ensemble_size"] = ensemble_size
    file["output_dim"] = output_dim
    file["T_std"] = T_std
    file["S_std"] = S_std
    file["latitude_range"] = latitude_range
    file["z_min"] = z_min
end

model_interface = joinpath(@__DIR__, "model_interface.jl")

# Include model interface so observation_map is available to the controller
include(model_interface)

@info "Calibration configuration:"
@info "  Number of parameters: 5"
@info "  Ensemble size: $ensemble_size"
@info "  Number of batches (nodes per iteration): $n_batches"
@info "  Number of iterations: $n_iterations"
@info "  Latitude ranges (extratropics): $latitude_range"
@info "  Depth range: z >= $z_min m"
@info "  T_std: $T_std, S_std: $S_std"
@info "  Output directory: $output_dir"

# ============================================
# HPC Configuration for GCP a3mega
# ============================================
hpc_kwargs = Dict(
    :time => 5 * 24 * 60,               # 1 hour (dry run is much shorter)
    :partition => "a3mega",
    :exclusive => true,        # Get exclusive access to entire node (all 8 GPUs)
)

# ============================================
# Run Calibration
# ============================================
@info "Starting CATKE calibration with BatchedSlurmGCPBackend..."
@info "Model interface: $model_interface"

ClimaCalibrate.calibrate(
    BatchedSlurmGCPBackend,
    ekp,
    n_iterations,
    priors,
    output_dir;
    model_interface = model_interface,
    hpc_kwargs = hpc_kwargs,
    verbose = true,
)

@info "Calibration completed!"
@info "Results saved to: $output_dir"