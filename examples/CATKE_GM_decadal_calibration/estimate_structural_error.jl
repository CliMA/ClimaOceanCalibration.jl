# Structural error of each field of the targets from the part of the residual that no parameter can change. The
# differences between the paired members of an iteration that both finished span the response of the observations to
# the parameters, and the residual of the central member outside that span sets the structural variance. The targets with these
# errors go to a new file, to start a calibration in a new directory with
# `calibrate_catke_gm_decadal.jl --targets <file> --restart-from <output directory>`.
#
#   julia --project examples/CATKE_GM_decadal_calibration/estimate_structural_error.jl <output directory> <iteration>

using ClimaCalibrate
using EnsembleKalmanProcesses
using LinearAlgebra

include("data_processing_decadal.jl")

output_directory, iteration = ARGS[1], parse(Int, ARGS[2])

targets = load(joinpath(@__DIR__, "calibration_data", "decadal_targets.jld2"), "targets")
ekp = ClimaCalibrate.load_ekp_struct(output_directory, iteration + 1)
G = get_g(ekp, iteration)

p = (size(G, 2) - 1) ÷ 2
differences = G[:, 2:p + 1] .- G[:, p + 2:2p + 1]
residual = get_obs(ekp) .- G[:, 1]
response = Matrix(qr(differences[:, findall(all(isfinite, differences; dims = 1)[:])]).Q)
unexplained = residual .- response * (response' * residual)

structural = map(targets, field_ranges(targets)) do field, range
    merge(field, (; structural_error = sqrt(mean(abs2, unexplained[range]))))
end

for name in keys(targets)
    @info "$name" previous = targets[name].structural_error estimated = structural[name].structural_error
end

jldsave(joinpath(@__DIR__, "calibration_data", "decadal_targets_structural.jld2"); targets = structural)
