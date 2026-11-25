# ClimaCalibrate interface of the decadal CATKE and Gent-McWilliams calibration.

using ClimaCalibrate
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using JLD2
using LinearAlgebra
using Statistics
using TOML

include("forward_model_onedegree.jl")
include("data_processing_decadal.jl")

"""
    CATKEGMDecadalInterface(output_directory, targets_file, ensemble_size, spinup_years, calibration_years)

The calibration in `output_directory` against the targets of `targets_file`, whose members run `spinup_years` and
then `calibration_years`.
"""
struct CATKEGMDecadalInterface <: ClimaCalibrate.AbstractModelInterface
    output_directory :: String
    targets_file :: String
    ensemble_size :: Int
    spinup_years :: Int
    calibration_years :: Int
end

ClimaCalibrate.model_interface_filepath(::CATKEGMDecadalInterface) = abspath(@__FILE__)
ClimaCalibrate.experiment_dir(::CATKEGMDecadalInterface) = abspath(joinpath(@__DIR__, "..", ".."))

member_checkpoints(directory) = filter(contains("checkpoint_iteration"), readdir(directory; join = true))

"""
    forward_model(interface::CATKEGMDecadalInterface, iteration, member)

Run the member with the scalings of its parameter file and save the vector it gives for the targets to
`observations.jld2`. Does nothing for a member that has that file.
"""
function ClimaCalibrate.forward_model(interface::CATKEGMDecadalInterface, iteration, member)
    directory = ClimaCalibrate.path_to_ensemble_member(interface.output_directory, iteration, member)
    observations = joinpath(directory, "observations.jld2")
    isfile(observations) && return nothing

    parameters = TOML.parsefile(ClimaCalibrate.parameter_path(interface.output_directory, iteration, member))
    scalings = Dict(name => parameter["value"] for (name, parameter) in parameters)
    run_decadal_member(scalings; directory, interface.spinup_years, interface.calibration_years)

    partial = joinpath(directory, "observations_partial.jld2")
    targets = load(interface.targets_file, "targets")
    jldsave(partial; G = member_observations(directory, targets; years = interface.calibration_years))
    mv(partial, observations)
    foreach(rm, member_checkpoints(directory))

    return nothing
end

"""
    observation_map(interface::CATKEGMDecadalInterface, iteration)

The vectors of the members of `iteration` as columns, NaN for a member without `observations.jld2`.
"""
function ClimaCalibrate.observation_map(interface::CATKEGMDecadalInterface, iteration)
    targets = load(interface.targets_file, "targets")
    G = fill(NaN, sum(length(field.y) for field in targets), interface.ensemble_size)

    for member in 1:interface.ensemble_size
        observations = joinpath(ClimaCalibrate.path_to_ensemble_member(interface.output_directory, iteration, member), "observations.jld2")
        isfile(observations) && (G[:, member] .= load(observations, "G"))
    end

    return G
end

"""
    derived_quantities(catke)

The turbulent Prandtl number `Pr`, the Schmidt number `Sc` of the turbulent kinetic energy and the mixing
coefficient `Γ` of `catke` on each plateau of its stability functions (negative, near-neutral and high Richardson
number, and convection), with the `steady_richardson_number` of sheared turbulence and the near-surface similarity
constant `κ` (Wagner et al., 2025).
"""
function derived_quantities(catke)
    ℓ = catke.mixing_length
    D = catke.turbulent_kinetic_energy_equation

    return (; Pr = (negative = ℓ.Cᵘⁿu / ℓ.Cᵘⁿc, neutral = ℓ.Cˡᵒu / ℓ.Cˡᵒc, high = ℓ.Cʰⁱu / ℓ.Cʰⁱc, convection = ℓ.Cᶜu / ℓ.Cᶜc),
              Sc = (negative = ℓ.Cᵘⁿu / ℓ.Cᵘⁿe, neutral = ℓ.Cˡᵒu / ℓ.Cˡᵒe, high = ℓ.Cʰⁱu / ℓ.Cʰⁱe, convection = ℓ.Cᶜu / ℓ.Cᶜe),
              Γ = (neutral = ℓ.Cˡᵒc / D.CˡᵒD, high = ℓ.Cʰⁱc / D.CʰⁱD),
              steady_richardson_number = ℓ.Cˡᵒu / (ℓ.Cˡᵒc + D.CˡᵒD),
              κ = ℓ.Cˢ * (ℓ.Cˡᵒu^3 / D.CˡᵒD)^(1 / 4))
end

"""
    analyze_iteration(interface::CATKEGMDecadalInterface, ekp, G, prior, output_directory, iteration)

Delete the checkpoints of the members of `iteration` and save its diagnostics to `diagnostics.jld2` in its directory:

- `misfit`: mean squared residual of each field and member, in units of its observational and structural variance;
- `mean`, `lower`, `upper`: the scalings at the updated mean and one standard deviation either side of it;
- `near_bound`: the scalings whose mean is above 95% of the upper bound of the prior, or less than 5% of the way from
  its lower bound to 1;
- `correlation`: correlation of the updated covariance of the unconstrained parameters;
- `sensitivity`: change of the misfit of each field per standard deviation of the prior of each parameter, from the
  pairs of members that both have a vector;
- `derived`: the `derived_quantities` at the updated mean;
- `failed`: the members without a vector, and `failed_scalings`, their scalings.
"""
function ClimaCalibrate.analyze_iteration(interface::CATKEGMDecadalInterface, ekp, G, prior, output_directory, iteration)
    iteration_directory = ClimaCalibrate.path_to_iteration(output_directory, iteration)
    for member in 1:interface.ensemble_size
        foreach(rm, member_checkpoints(ClimaCalibrate.path_to_ensemble_member(output_directory, iteration, member)))
    end

    targets = load(interface.targets_file, "targets")
    names = get_name(prior)
    U = get_u(ekp, iteration)
    scalings = transform_unconstrained_to_constrained(prior, U)
    constrained(u) = transform_unconstrained_to_constrained(prior, u)

    misfit = map(targets, field_ranges(targets)) do field, range
        vec(mean(abs2.(field.y .- G[range, :]) ./ (field.observation_error .^ 2 .+ field.structural_error^2); dims = 1))
    end

    u = get_u_mean_final(ekp)
    covariance = get_u_cov_final(ekp)
    σ = sqrt.(diag(covariance))
    scaling_mean = constrained(u)
    lower, upper = constrained(fill(-50.0, length(u))), constrained(fill(50.0, length(u)))
    near_bound = names[@. (scaling_mean > 0.95 * upper) | (scaling_mean - lower < 0.05 * (1 - lower))]

    failed = findall(member -> any(isnan, view(G, :, member)), 1:interface.ensemble_size)

    p = length(u)
    pairs = filter(i -> isdisjoint((1 + i, 1 + p + i), failed), 1:p)
    plus, minus = 1 .+ pairs, 1 + p .+ pairs
    misfit_difference = stack(collect(field[plus] .- field[minus]) for field in misfit; dims = 1)
    sensitivity = misfit_difference / (U[:, plus] .- U[:, minus]) .* sqrt.(diag(cov(prior)))'

    jldsave(joinpath(iteration_directory, "diagnostics.jld2"); names, misfit, sensitivity, near_bound,
            mean = scaling_mean, lower = constrained(u .- σ), upper = constrained(u .+ σ),
            correlation = covariance ./ (σ .* σ'),
            derived = derived_quantities(calibrated_catke(Dict(names .=> scaling_mean))),
            failed, failed_scalings = scalings[:, failed])

    @info "Iteration $iteration" misfit_of_mean = map(first, misfit) scalings = Dict(names .=> scaling_mean) near_bound failed

    return nothing
end
