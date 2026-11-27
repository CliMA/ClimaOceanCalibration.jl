# Structural error of each field of the targets: the part of the residual of the central member of an iteration that
# no linear combination of the responses of its paired members removes, with the values weighted as in the misfit,
# less the observation variance. It only raises the structural error of a field. The targets with these errors go to
# `calibration_data/decadal_targets_structural.jld2`, to start a calibration in a new directory with
# `calibrate_catke_gm_decadal.jl --targets <file>`.
#
#   julia --project examples/CATKE_GM_decadal_calibration/estimate_structural_error.jl <output directory> <iteration>

using ClimaCalibrate
using LinearAlgebra

include("data_processing_decadal.jl")

output_directory, iteration = ARGS[1], parse(Int, ARGS[2])

targets = load(joinpath(@__DIR__, "calibration_data", "decadal_targets.jld2"), "targets")
redundancy = load(joinpath(@__DIR__, "calibration_data", "redundancy.jld2"), "redundancy")
ranges = field_ranges(targets)
y = reduce(vcat, [field.y for field in targets])

iteration_directory = ClimaCalibrate.path_to_iteration(output_directory, iteration)
ensemble_size = count(startswith("member_"), readdir(iteration_directory))
p = (ensemble_size - 1) ÷ 2

function member_vector(member)
    path = joinpath(ClimaCalibrate.path_to_ensemble_member(output_directory, iteration, member), "observations.jld2")
    return isfile(path) ? load(path, "G") : fill(NaN, length(y))
end

G = stack(member_vector(m) for m in 1:ensemble_size)
differences = G[:, 2:p + 1] .- G[:, p + 2:2p + 1]
differences = differences[:, vec(all(isfinite, differences; dims = 1))]
residual = y .- G[:, 1]

weight = sqrt.(reduce(vcat, [1 ./ (redundancy[name] .* (field.observation_error .^ 2 .+ field.structural_error^2))
                             for (name, field) in pairs(targets)]))
response = Matrix(qr(weight .* differences).Q)
unexplained = (weight .* residual .- response * (response' * (weight .* residual))) ./ weight

structural = map(targets, ranges) do field, range
    estimate = sqrt(max(0, mean(abs2, unexplained[range]) - mean(abs2, field.observation_error)))
    merge(field, (; structural_error = max(field.structural_error, estimate)))
end

for name in keys(targets)
    @info "$name" previous = targets[name].structural_error estimated = structural[name].structural_error
end

jldsave(joinpath(@__DIR__, "calibration_data", "decadal_targets_structural.jld2"); targets = structural)
