# calibrate_catke_gm.jl
#
# Main calibration driver for CATKE + GM scaling parameters against WOA on
# the ORCA grid. Uses ClimaCalibrate v0.3.0 AbstractModelInterface + HPCBackend
# instance APIs.
#
# Forward model: 10-year orca OMIP run with the omip_simulation(:orca; ...)
# physics from examples/OMIP_GCP/orca_corrected_snow_kskew1000_ksymm1000_bih50days_10yr.jl.
#
# Observation target: WOA annual T,S on the ORCA grid, tropical band
# (-20°..20°), upper 200 m, [T..., S...] concatenated.

using ClimaCalibrate
isdefined(ClimaCalibrate, :Backend) || error(
    "calibrate_catke_gm.jl requires ClimaCalibrate v0.3.x; " *
    "the active project appears to resolve an older ClimaCalibrate."
)
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
            help = "Enable the GM (IsopycnalSkewSymmetric) eddy closure in the forward model"
            arg_type = Bool
            default = true
        "--calibrate_gm"
            help = "Also calibrate the three GM scaling parameters (κ_skew, κ_symmetric, max_slope)"
            arg_type = Bool
            default = false
        "--SKIN_TEMPERATURE"
            help = "Compute the atmosphere-ocean interface temperature as a flux-balance skin temperature (SkinTemperature) instead of the bulk top-cell temperature"
            arg_type = Bool
            default = false
        "--DZ_TOP"
            help = "Surface-cell thickness in metres for the forward-model vertical grid. Pass `false` (the default) to use the omip_simulation default grid."
            arg_type = String
            default = "false"
    end
    return parse_args(s)
end

const args             = parse_commandline()
const use_gm           = args["GM"]
const calibrate_gm     = args["calibrate_gm"]
const skin_temperature = args["SKIN_TEMPERATURE"]

# DZ_TOP=false (or nothing/default) ⇒ use the omip_simulation default grid (Δz_top = nothing);
# otherwise parse the surface-cell thickness in metres.
const Δz_top = let v = lowercase(strip(args["DZ_TOP"]))
    (v in ("false", "nothing", "default")) ? nothing : parse(Float64, v)
end

# ============================================
# Configuration
# ============================================
const n_iterations      = 10                 # number of EKI iterations
const prior_std         = 0.5                # stddev of each scaling prior (mean = 1, bounded > 0)
const T_std             = 0.2                # observation noise stddev for T (°C); sets diagonal cov entry T_std^2
const S_std             = T_std / 4          # observation noise stddev for S (PSU); ratio 1:4 matches typical T,S scale
const simulation_length = 5                  # forward-model run length in years
const sampling_length   = 3                  # years averaged at the end (the calibration target window: years simulation_length-sampling_length .. simulation_length)
const latitude_range    = (-20.0, 20.0)      # tropical band compared against WOA (degrees latitude)
const z_min             = -200.0             # upper-ocean depth cutoff in metres; only cells with z ≥ z_min are in the loss
const filename_prefix = "orca_catke_gm_calibration"  # prefix for forward model output files
const staging_dir = nothing                 # Disable per-member JRA55 staging by default.
const with_ice_dynamics = false             # disable sea-ice dynamics in the OMIP forward model

# Calibrated parameter names. These keys must match those produced by
# CATKE_SCALING_SPEC / GM_SCALING_SPEC in forward_model_orca.jl. Comment
# out a name to drop it from the calibration (then it stays fixed at 1).
# All 18 CATKEMixingLength + 7 CATKEEquation non-zero coefficients are
# included by default. Cᵉu, Cᵉe, CᵉD are omitted because they default to 0
# (scaling is degenerate).
const catke_param_names = (
    # CATKEMixingLength
    "Cˢ_scaling",    # Surface distance coefficient for shear length scale
    # "Cᵇ_scaling",    # Bottom distance coefficient for shear length scale
    # "Cˢᵖ_scaling",   # Sheared convective plume coefficient

    "CRiᵟ_scaling",  # Stability function width
    "CRi⁰_scaling",  # Stability function lower Ri

    "Cʰⁱu_scaling",  # Shear mixing length coefficient for momentum at high Ri
    "Cˡᵒu_scaling",  # Shear mixing length coefficient for momentum at low Ri
    # "Cᵘⁿu_scaling",  # Shear mixing length coefficient for momentum at negative Ri
    # "Cᶜu_scaling",   # Convective mixing length coefficient for momentum

    "Cʰⁱc_scaling",  # Shear mixing length coefficient for tracers at high Ri
    "Cˡᵒc_scaling",  # Shear mixing length coefficient for tracers at low Ri
    # "Cᵘⁿc_scaling",  # Shear mixing length coefficient for tracers at negative Ri
    # "Cᶜc_scaling",   # Convective mixing length coefficient for tracers
    # "Cᵉc_scaling",   # Convective penetration mixing length coefficient for tracers

    "Cʰⁱe_scaling",  # Shear mixing length coefficient for TKE at high Ri
    "Cˡᵒe_scaling",  # Shear mixing length coefficient for TKE at low Ri
    # "Cᵘⁿe_scaling",  # Shear mixing length coefficient for TKE at negative Ri
    # "Cᶜe_scaling",   # Convective mixing length coefficient for TKE

    # CATKEEquation
    "CʰⁱD_scaling",  # Dissipation length scale shear coefficient for high Ri
    "CˡᵒD_scaling",  # Dissipation length scale shear coefficient for low Ri
    # "CᵘⁿD_scaling",  # Dissipation length scale shear coefficient for negative Ri
    # "CᶜD_scaling",   # Dissipation length scale convecting layer coefficient
    # "Cᵂu★_scaling",  # Surface shear-driven TKE flux coefficient
    # "CᵂwΔ_scaling",  # Surface convective TKE flux coefficient
    # "Cᵂϵ_scaling",   # Dissipative near-bottom TKE flux coefficient
)

# GM scaling parameters, only calibrated when --calibrate_gm is true.
const gm_param_names = calibrate_gm ? (
    "κ_skew_scaling",       # GM skew diffusivity
    "κ_symmetric_scaling",  # Redi symmetric diffusivity
    "max_slope_scaling",    # FluxTapering slope limiter max slope
) : ()

# The WOA target must live on the SAME vertical grid as the forward model, since
# the comparison does no vertical regridding. Δz_top = nothing ⇒ default grid;
# Δz_top = h ⇒ the woa_orca_grid_dztop<h>.jld2 cache built by precompute_woa_orca.jl.
const woa_file = abspath(joinpath(@__DIR__, "calibration_data",
    Δz_top === nothing ? "woa_orca_grid.jld2" : "woa_orca_grid_dztop$(Δz_top).jld2"))
isfile(woa_file) || error("""
    WOA-on-ORCA cache not found at:
        $woa_file
    Run precompute_woa_orca.jl first (with ΔZ_TOP = $(Δz_top === nothing ? "nothing" : Δz_top)).
""")

# Output directory (defined BEFORE model_interface.jl is included, since the
# legacy model_interface.jl that it loads requires `output_dir` to be a global.)
output_dir = joinpath(pwd(), "calibration_runs",
    "catke_$(length(catke_param_names))_gm_$(length(gm_param_names))_prior_$(prior_std)_Tstd_$(T_std)_Sstd_$(S_std)" *
    "_simlength_$(simulation_length)yr_samplength_$(sampling_length)yr_lat$(latitude_range[2])_zmin$(z_min)_gm$(use_gm)" *
    "_dz$(Δz_top === nothing ? "default" : Δz_top)" *
    (skin_temperature ? "_skintemp" : ""))

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
                            # TransformUnscented(priors, sigma_points = "simplex");
                            TransformUnscented(priors);
                            scheduler)

const ensemble_size = EnsembleKalmanProcesses.get_N_ens(ekp)
n_batches = ceil(Int, ensemble_size / 8)

# Guard against resuming into a directory whose stored config is incompatible
# with the current one. ClimaCalibrate.calibrate reconstructs the EKP from the
# on-disk iteration files when resuming, so a mismatch in output_dim (e.g. a
# different Δz_top → different number of WOA cells) silently survives until
# update_ensemble! tries to broadcast the new G against the stale observation
# and dies with a DimensionMismatch. Catch it here, before any jobs are submitted.
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
        Refusing to resume: existing calibration state in
            $output_dir
        was created with an incompatible configuration:
            $(join(mismatches, "\n            "))
        Resuming would feed the new forward-model output into the stale EKP
        observation and crash in update_ensemble!. Use a fresh output_dir, or
        remove/rename the existing one to start from iteration 0.
    """)
end

# Persist metadata that worker processes will read from
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
    file["catke_param_names"]  = collect(catke_param_names)
    file["gm_param_names"]     = collect(gm_param_names)
    file["use_gm"]             = use_gm
    file["with_ice_dynamics"]  = with_ice_dynamics
    file["Δz_top"]             = Δz_top
    file["skin_temperature"]   = skin_temperature
end

# v0.3.0 backend + interface
include(joinpath(@__DIR__, "batched_slurm_backend.jl"))
include(joinpath(@__DIR__, "model_interface.jl"))

interface = CATKEGMInterface()
@info "Model interface: $(ClimaCalibrate.model_interface_filepath(interface))"
@info "Experiment dir:  $(ClimaCalibrate.experiment_dir(interface))"

@info "Calibration configuration:"
@info "  use_gm:            $use_gm"
@info "  with_ice_dynamics: $with_ice_dynamics"
@info "  skin_temperature:  $skin_temperature"
@info "  Δz_top:            $(Δz_top === nothing ? "default" : Δz_top)"
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

backend = BatchedSlurmGCPBackend(slurm_cfg)

@info "Starting calibration with BatchedSlurmGCPBackend..."
ClimaCalibrate.calibrate(
    backend,
    ekp,
    interface,
    n_iterations,
    priors,
    output_dir,
)

@info "Calibration completed. Results: $output_dir"
