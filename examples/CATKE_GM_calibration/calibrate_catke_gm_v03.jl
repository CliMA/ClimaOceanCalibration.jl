# calibrate_catke_gm_v03.jl
#
# v0.3.0-compatible driver for the ORCA CATKE+GM calibration.
# Parallels calibrate_catke_gm.jl but uses the new ClimaCalibrate v0.3.0
# AbstractModelInterface + HPCBackend instance APIs.
#
# Forward model: 10-year orca OMIP run with the omip_simulation(:orca; ...)
# physics from examples/OMIP_GCP/orca_corrected_snow_kskew1000_ksymm1000_bih50days_10yr.jl.
#
# Observation target: WOA annual T,S on the ORCA grid, tropical band
# (-20°..20°), upper 200 m, [T..., S...] concatenated.

using ClimaCalibrate
using ClimaCalibrate.Backend: SlurmConfig
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using JLD2

# ============================================
# Configuration
# ============================================
const n_iterations      = 10
const prior_std         = 0.2
const T_std             = 0.2
const S_std             = T_std / 4
const simulation_length = 10
const sampling_length   = 5
const latitude_range    = (-20.0, 20.0)
const z_min             = -200.0
const filename_prefix   = "orca_catke_gm_calibration"

const catke_param_names = (
    "Cˢ_scaling",
    "CRiᵟ_scaling",
    "CRi⁰_scaling",
    "Cʰⁱu_scaling",
    "Cˡᵒu_scaling",
    "Cʰⁱc_scaling",
    "Cˡᵒc_scaling",
    "Cʰⁱe_scaling",
    "Cˡᵒe_scaling",
    "CʰⁱD_scaling",
    "CˡᵒD_scaling",
)

const gm_param_names = ()

const woa_file = abspath(joinpath(@__DIR__, "calibration_data", "woa_orca_grid.jld2"))
isfile(woa_file) || error("""
    WOA-on-ORCA cache not found at:
        $woa_file
    Run precompute_woa_orca.jl first.
""")

# Output directory (defined BEFORE model_interface_v03.jl is included, since the
# legacy model_interface.jl that it loads requires `output_dir` to be a global.)
output_dir = joinpath(pwd(), "calibration_runs",
    "catke_$(length(catke_param_names))_gm_$(length(gm_param_names))_prior_$(prior_std)_Tstd_$(T_std)_Sstd_$(S_std)" *
    "_simlength_$(simulation_length)yr_samplength_$(sampling_length)yr_lat$(latitude_range[2])_zmin$(z_min)_v03")

mkpath(output_dir)

# ============================================
# Priors
# ============================================
function make_prior(name)
    return constrained_gaussian(name, 1.0, prior_std, 0, Inf)
end

priors = combine_distributions(
    vcat(
        [make_prior(n) for n in catke_param_names],
        [make_prior(n) for n in gm_param_names],
    )
)

# ============================================
# Observation target + covariance
# ============================================
# data_processing.jl is included transitively via model_interface.jl below, but
# we need process_woa_target / build_diagonal_covariance now (before the
# include), so pull it in directly.
include(joinpath(@__DIR__, "data_processing.jl"))

@info "Building WOA target from $woa_file (lat $latitude_range, z ≥ $z_min)..."
Y_target = process_woa_target(woa_file; lat_range = latitude_range, z_min)

@info "Building diagonal covariance: T_std=$T_std, S_std=$S_std"
covariance = build_diagonal_covariance(Y_target;
                                        T_variance = T_std^2,
                                        S_variance = S_std^2)

const output_dim = length(Y_target)
@info "Output dimension: $output_dim"

Y_obs = Observation(Dict(
    "samples"     => Y_target,
    "covariances" => covariance,
    "names"       => "WOA_annual_ORCA_tropics_top$(abs(z_min))",
))

# ============================================
# EKP
# ============================================
scheduler = DataMisfitController(on_terminate = "continue")
@info "Creating EnsembleKalmanProcess..."
ekp = EnsembleKalmanProcess(Y_obs,
                            TransformUnscented(priors, sigma_points = "simplex");
                            scheduler)

const ensemble_size = EnsembleKalmanProcesses.get_N_ens(ekp)
n_batches = ceil(Int, ensemble_size / 8)

# Persist metadata that worker processes will read from
jldopen(joinpath(output_dir, "calibration_metadata.jld2"), "w") do file
    file["ensemble_size"]      = ensemble_size
    file["output_dim"]         = output_dim
    file["latitude_range"]     = latitude_range
    file["z_min"]              = z_min
    file["T_std"]              = T_std
    file["S_std"]              = S_std
    file["simulation_length"]  = simulation_length
    file["sampling_length"]    = sampling_length
    file["filename_prefix"]    = filename_prefix
    file["woa_file"]           = woa_file
    file["catke_param_names"]  = collect(catke_param_names)
    file["gm_param_names"]     = collect(gm_param_names)
end

# v0.3.0 backend + interface
include(joinpath(@__DIR__, "batched_slurm_backend_v03.jl"))
include(joinpath(@__DIR__, "model_interface_v03.jl"))

interface = CATKEGMInterface()
@info "Model interface: $(ClimaCalibrate.model_interface_filepath(interface))"
@info "Experiment dir:  $(ClimaCalibrate.experiment_dir(interface))"

@info "Calibration configuration:"
@info "  CATKE params: $(catke_param_names)"
@info "  GM params:    $(gm_param_names)"
@info "  Ensemble size: $ensemble_size  ($n_batches a3mega nodes per iteration)"
@info "  Iterations:    $n_iterations"
@info "  Latitude:      $latitude_range"
@info "  Depth:         z ≥ $z_min m"
@info "  T_std=$T_std  S_std=$S_std  prior_std=$prior_std"
@info "  Output dir:    $output_dir"

# ============================================
# HPC config (GCP a3mega, 8 H100s per node)
# ============================================
slurm_cfg = SlurmConfig(
    directives = [
        :time      => 5 * 24 * 60,    # minutes; SlurmConfig formats to DD-HH:MM:SS
        :partition => "a3mega",
        :exclusive => "user",          # `--exclusive=user` is the documented form
    ],
)

backend = BatchedSlurmGCPBackendV03(slurm_cfg)

@info "Starting calibration with BatchedSlurmGCPBackendV03..."
ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    priors,
    output_dir,
)

@info "Calibration completed. Results: $output_dir"
