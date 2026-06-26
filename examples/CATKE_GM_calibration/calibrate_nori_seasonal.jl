# calibrate_nori_seasonal.jl
#
# Seasonal-cycle calibration driver for the NORi Richardson-number-based
# vertical diffusivity closure. Calibrates three NORi parameters:
#   νˢʰ, Pr_shearₜ, Riᶜ
# against the WOA-monthly zonal-mean seasonal cycle on the ORCA grid,
# tropical band (−20°..20°), upper 200 m.
#
# Follows the same pattern as calibrate_catke_gm_seasonal.jl. Run
# precompute_woa_monthly_zonal.jl first to build the WOA seasonal target.

using ClimaCalibrate
isdefined(ClimaCalibrate, :Backend) || error(
    "calibrate_nori_seasonal.jl requires ClimaCalibrate v0.3.x; " *
    "the active project appears to resolve an older ClimaCalibrate.")
using ClimaCalibrate.Backend: SlurmConfig
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using JLD2
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--GM"
            help = "Enable the GM (IsopycnalSkewSymmetric) eddy closure"
            arg_type = Bool
            default = true
        "--calibrate_gm"
            help = "Also calibrate the three GM scaling parameters (κ_skew, κ_symmetric, max_slope)"
            arg_type = Bool
            default = false
        "--SKIN_TEMPERATURE"
            help = "Compute the atmosphere-ocean interface temperature as a flux-balance skin temperature"
            arg_type = Bool
            default = false
        "--DZ_TOP"
            help = "Surface-cell thickness in metres. Pass `false` (default) for the omip_simulation default grid."
            arg_type = String
            default = "false"
    end
    return parse_args(s)
end

const args             = parse_commandline()
const use_gm           = args["GM"]
const calibrate_gm     = args["calibrate_gm"]
const skin_temperature = args["SKIN_TEMPERATURE"]

const Δz_top = let v = lowercase(strip(args["DZ_TOP"]))
    (v in ("false", "nothing", "default")) ? nothing : parse(Float64, v)
end

# ============================================
# Configuration
# ============================================
const n_iterations      = 10
const prior_std         = 0.5
const T_std             = 0.2                # observation noise stddev for T (°C)
const S_std             = T_std / 4          # observation noise stddev for S (PSU)
const simulation_length = 5                  # forward-model run length in years
const sampling_length   = 3                  # years for the legacy end-of-run mean writer
const latitude_range    = (-20.0, 20.0)
const z_min             = -200.0
const filename_prefix   = "orca_nori_seasonal"
const staging_dir       = nothing
const with_ice_dynamics = false

# Three NORi parameters to calibrate (as scaling factors on their calibrated defaults).
const nori_param_names = (
    "νˢʰ_scaling",
    "Pr_shearₜ_scaling",
    "Riᶜ_scaling",
)

const gm_param_names = calibrate_gm ? (
    "κ_skew_scaling",
    "κ_symmetric_scaling",
    "max_slope_scaling",
) : ()

# Precomputed WOA-monthly zonal target (must match Δz_top).
const woa_file = abspath(joinpath(@__DIR__, "calibration_data",
    Δz_top === nothing ? "woa_monthly_zonal.jld2" : "woa_monthly_zonal_dztop$(Δz_top).jld2"))
isfile(woa_file) || error("""
    WOA monthly-zonal target not found at:
        $woa_file
    Run precompute_woa_monthly_zonal.jl first (with DZ_TOP = $(Δz_top === nothing ? "false" : Δz_top)).
""")

output_dir = joinpath(pwd(), "calibration_runs",
    "seasonal_nori_$(length(nori_param_names))_gm_$(length(gm_param_names))_prior_$(prior_std)" *
    "_Tstd_$(T_std)_Sstd_$(S_std)_simlength_$(simulation_length)yr_lat$(latitude_range[2])_zmin$(z_min)" *
    "_gm$(use_gm)_dz$(Δz_top === nothing ? "default" : Δz_top)" *
    (skin_temperature ? "_skintemp" : ""))

mkpath(output_dir)

# ============================================
# Priors
# ============================================
make_prior(name) = constrained_gaussian(name, 1.0, prior_std, 0, Inf)

priors = combine_distributions(
    vcat(
        [make_prior(n) for n in nori_param_names],
        [make_prior(n) for n in gm_param_names],
    )
)

# ============================================
# Observation target + covariance
# ============================================
include(joinpath(@__DIR__, "data_processing_seasonal.jl"))

@info "Loading WOA monthly-zonal seasonal target from $woa_file..."
Y_target = load_woa_seasonal_target(woa_file)

@info "Building seasonal diagonal covariance: T_std=$T_std, S_std=$S_std"
covariance = build_seasonal_covariance(Y_target; T_variance = T_std^2, S_variance = S_std^2)

const output_dim = length(Y_target)
@info "Seasonal output dimension: $output_dim (12 months × tropical/upper-200 m T+S cells)"

Y_obs = Observation(Dict(
    "samples"     => Y_target,
    "covariances" => covariance,
    "names"       => "WOA_monthly_zonal_ORCA_tropics_top$(abs(z_min))",
))

# ============================================
# EKP
# ============================================
scheduler = DataMisfitController(on_terminate = "continue")
@info "Creating EnsembleKalmanProcess..."
ekp = EnsembleKalmanProcess(Y_obs,
                            TransformUnscented(priors);
                            scheduler)

const ensemble_size = EnsembleKalmanProcesses.get_N_ens(ekp)
n_batches = ceil(Int, ensemble_size / 8)

# Resume-compatibility guard.
metadata_path = joinpath(output_dir, "calibration_metadata.jld2")
if isfile(metadata_path)
    prev = jldopen(metadata_path, "r") do file
        (output_dim = haskey(file, "output_dim") ? file["output_dim"] : nothing,
         Δz_top     = haskey(file, "Δz_top")     ? file["Δz_top"]     : nothing,
         woa_file   = haskey(file, "woa_file")   ? file["woa_file"]   : nothing)
    end
    mismatches = String[]
    prev.output_dim === nothing || prev.output_dim == output_dim ||
        push!(mismatches, "output_dim: stored $(prev.output_dim) ≠ current $output_dim")
    prev.Δz_top === nothing && Δz_top === nothing || isequal(prev.Δz_top, Δz_top) ||
        push!(mismatches, "Δz_top: stored $(prev.Δz_top) ≠ current $Δz_top")
    prev.woa_file === nothing || prev.woa_file == woa_file ||
        push!(mismatches, "woa_file: stored $(prev.woa_file) ≠ current $woa_file")
    isempty(mismatches) || error("""
        Refusing to resume: existing NORi seasonal calibration state in
            $output_dir
        was created with an incompatible configuration:
            $(join(mismatches, "\n            "))
        Use a fresh output_dir, or remove/rename the existing one.
    """)
end

jldopen(metadata_path, "w") do file
    file["ensemble_size"]      = ensemble_size
    file["output_dim"]         = output_dim
    file["latitude_range"]     = latitude_range
    file["z_min"]              = z_min
    file["T_std"]              = T_std
    file["S_std"]              = S_std
    file["simulation_length"]  = simulation_length
    file["sampling_length"]    = sampling_length
    file["filename_prefix"]    = filename_prefix
    file["staging_dir"]        = staging_dir
    file["woa_file"]           = woa_file
    file["nori_param_names"]   = collect(nori_param_names)
    file["gm_param_names"]     = collect(gm_param_names)
    file["use_gm"]             = use_gm
    file["with_ice_dynamics"]  = with_ice_dynamics
    file["Δz_top"]             = Δz_top
    file["skin_temperature"]   = skin_temperature
    file["output_mode"]        = "seasonal"
    file["n_months"]           = SEASONAL_N_MONTHS
end

include(joinpath(@__DIR__, "batched_slurm_backend.jl"))
include(joinpath(@__DIR__, "model_interface_nori_seasonal.jl"))

interface = NORiSeasonalInterface()
@info "Model interface: $(ClimaCalibrate.model_interface_filepath(interface))"
@info "Experiment dir:  $(ClimaCalibrate.experiment_dir(interface))"

@info "NORi seasonal calibration configuration:"
@info "  use_gm:            $use_gm"
@info "  with_ice_dynamics: $with_ice_dynamics"
@info "  skin_temperature:  $skin_temperature"
@info "  Δz_top:            $(Δz_top === nothing ? "default" : Δz_top)"
@info "  NORi params: $(nori_param_names)"
@info "  GM params:   $(gm_param_names)"
@info "  Ensemble size: $ensemble_size  ($n_batches a3mega nodes per iteration)"
@info "  Iterations:    $n_iterations"
@info "  Latitude:      $latitude_range   z ≥ $z_min m   (12-month seasonal cycle)"
@info "  T_std=$T_std  S_std=$S_std  prior_std=$prior_std"
@info "  Output dir:    $output_dir"

# ============================================
# HPC config (GCP a3mega, 8 H100s per node)
# ============================================
slurm_cfg = SlurmConfig(
    directives = [
        :time      => 5 * 24 * 60,
        :partition => "a3mega",
        :exclusive => "user",
    ],
)

backend = BatchedSlurmGCPBackend(slurm_cfg)

@info "Starting NORi seasonal calibration with BatchedSlurmGCPBackend..."
ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    priors,
    output_dir,
)

@info "NORi seasonal calibration completed. Results: $output_dir"
