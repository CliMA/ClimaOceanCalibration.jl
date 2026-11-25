# Calibration of scalings of the CATKE and Gent-McWilliams coefficients of ClimaOcean's `OneDegree()` configuration
# against the 2005–2014 climatologies of `precompute_targets_decadal.jl`, by unscented transform Kalman inversion.
# Members run from 1995 to 2014 and the last ten years enter the misfit.
#
#   julia --project examples/CATKE_GM_decadal_calibration/calibrate_catke_gm_decadal.jl --suite 26
#
# Run `precompute_targets_decadal.jl`, `run_default_member.jl` and `estimate_redundancy.jl` first.

using ArgParse
using ClimaCalibrate
using ClimaCalibrate.Backend: SlurmConfig
using ClimaOcean: OneDegree
using ConstructionBase: getproperties
using Distributions
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2
using LinearAlgebra

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--suite"
        help = "26: every CATKE coefficient of Wagner et al. (2025); 13: the coefficients of each regime share a scaling"
        arg_type = Int
        default = 26
        range_tester = in((13, 26))
    "--iterations"
        arg_type = Int
        default = 10
    "--output-directory"
        default = joinpath(pwd(), "calibration_runs", "catke_gm_decadal")
    "--targets"
        default = joinpath(@__DIR__, "calibration_data", "decadal_targets.jld2")
    "--redundancy"
        default = joinpath(@__DIR__, "calibration_data", "redundancy.jld2")
    "--restart-from"
        help = "Output directory of a calibration whose last mean and covariance start this one"
end
arguments = parse_args(settings)

include("model_interface_decadal.jl")
include(joinpath(@__DIR__, "..", "CATKE_GM_calibration", "batched_slurm_backend.jl"))

#####
##### Priors of the scalings
#####

# Upper bounds of the CATKE coefficients in the calibration of Wagner et al. (2025), Table 3
const upper_bounds = (Cᵂu★ = 8, CᵂwΔ = 8, Cˢ = 2, Cˢᵖ = 2, Cᵉc = 2, CRi⁰ = 2, CRiᵟ = 2,
                      Cᵘⁿu = 2, Cˡᵒu = 2, Cʰⁱu = 2, Cᶜu = 8,
                      Cᵘⁿc = 2, Cˡᵒc = 2, Cʰⁱc = 2, Cᶜc = 8,
                      Cᵘⁿe = 10, Cˡᵒe = 10, Cʰⁱe = 10, Cᶜe = 10,
                      CᵘⁿD = 10, CˡᵒD = 10, CʰⁱD = 10, CᶜD = 10)

const catke = OneDegree().ocean.vertical_mixing
const coefficients = merge(getproperties(catke.mixing_length), getproperties(catke.turbulent_kinetic_energy_equation))

"""
    scaling_prior(name, upper; lower = 0, σ = 0.5)

Prior of the scaling `name` between `lower` and `upper`: normal with standard deviation `σ` in the unconstrained
space, with median 1.
"""
scaling_prior(name, upper; lower = 0, σ = 0.5) =
    ParameterDistribution(Parameterized(Normal(log((1 - lower) / (upper - 1)), σ)), bounded(lower, upper), name)

# The lower bound of `CRi⁰` keeps the transition of the stability functions above the steady-state Richardson number
const lower_bounds = (; CRi⁰ = derived_quantities(catke).steady_richardson_number / coefficients.CRi⁰)

coefficient_prior(name) = scaling_prior("$(name)_scaling", upper_bounds[name] / coefficients[name];
                                        lower = get(lower_bounds, name, 0))

catke_priors = if arguments["suite"] == 26
    [coefficient_prior(name) for name in keys(upper_bounds)]
else
    [[scaling_prior("$(group)_scaling", 2.5) for group in ("Cᵘⁿ", "Cˡᵒʰⁱ", "Cᶜ")];
     [coefficient_prior(name) for name in (:Cˢ, :Cˢᵖ, :Cᵉc, :Cᵂu★, :CᵂwΔ, :CRi⁰, :CRiᵟ)]]
end

log_normal_prior(name, σ) = ParameterDistribution(Parameterized(Normal(0, σ)), bounded_below(0), name)

priors = combine_distributions([catke_priors;
                                log_normal_prior("κ_skew_scaling", 0.5);
                                log_normal_prior("κ_symmetric_scaling", 0.5);
                                log_normal_prior("max_slope_scaling", log(10) / 2)])

#####
##### Observations: the noise of each field is its observational and structural variance times its redundancy
#####

targets_file = abspath(arguments["targets"])
targets = load(targets_file, "targets")
redundancy = load(arguments["redundancy"], "redundancy")

noise(name, field) = Diagonal(redundancy[name] .* (field.observation_error .^ 2 .+ field.structural_error^2))

observations = combine_observations([Observation(Dict("samples" => field.y,
                                                      "covariances" => noise(name, field),
                                                      "names" => string(name)))
                                     for (name, field) in pairs(targets)])

#####
##### Unscented transform Kalman inversion
#####

process = if isnothing(arguments["restart-from"])
    TransformUnscented(priors; sigma_points = "symmetric", impose_prior = true)
else
    previous_directory = arguments["restart-from"]
    last_iteration = ClimaCalibrate.last_completed_iteration(previous_directory)
    previous = ClimaCalibrate.load_ekp_struct(previous_directory, last_iteration + 1)
    TransformUnscented(get_u_mean_final(previous), Matrix(get_u_cov_final(previous));
                       prior_mean = mean(priors), prior_cov = Matrix(cov(priors)),
                       sigma_points = "symmetric", impose_prior = true)
end

ekp = EnsembleKalmanProcess(observations, process;
                            scheduler = DataMisfitController(on_terminate = "continue"),
                            failure_handler_method = SampleSuccGauss())

output_directory = mkpath(abspath(arguments["output-directory"]))
metadata_file = joinpath(output_directory, "calibration_metadata.jld2")

fingerprint = hash((get_name(priors), mean(priors), cov(priors),
                    transform_unconstrained_to_constrained(priors, mean(priors) .+ 1),
                    get_u_final(ekp), get_obs(ekp), get_obs_noise_cov(ekp; build = false)))

if isfile(metadata_file)
    load(metadata_file, "fingerprint") == fingerprint ||
        error("$output_directory holds a calibration with other priors, observations or noise.")
else
    jldsave(metadata_file; fingerprint, arguments, climacalibrate_version = pkgversion(ClimaCalibrate))
end

interface = CATKEGMDecadalInterface(output_directory, targets_file, get_N_ens(ekp), 10, 10)

slurm = SlurmConfig(directives = [:time => 5 * 24 * 60, :partition => "a3mega", :exclusive => "user"])
backend = BatchedSlurmGCPBackend(slurm)

ClimaCalibrate.calibrate(backend, ekp, interface, arguments["iterations"], priors, output_directory)
