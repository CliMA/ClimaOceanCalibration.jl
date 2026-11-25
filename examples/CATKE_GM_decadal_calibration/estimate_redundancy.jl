# Redundancy of each field of the targets, the factor that inflates its noise, from the residual of the member that
# `run_default_member.jl` ran. The residual in units of its standard deviation has a correlation `ρ` between neighbors
# along each axis of the field, which leaves the fraction `min(1, -log(ρ))` of the values along it independent.
# Saved to `calibration_data/redundancy.jld2`.
#
#   julia --project examples/CATKE_GM_decadal_calibration/estimate_redundancy.jl <member directory>

include("data_processing_decadal.jl")

"""
    neighbor_correlation(z, dimension)

Correlation between the finite neighbors of `z` along `dimension`.
"""
function neighbor_correlation(z, dimension)
    N = size(z, dimension)
    near, far = selectdim(z, dimension, 1:N - 1), selectdim(z, dimension, 2:N)
    pairs = @. isfinite(near) & isfinite(far)
    return cor(near[pairs], far[pairs])
end

independent_fraction(ρ) = ρ > 0 ? min(1, -log(ρ)) : 1

targets = load(joinpath(@__DIR__, "calibration_data", "decadal_targets.jld2"), "targets")
G = member_observations(ARGS[1], targets)

redundancy = map(targets, field_ranges(targets)) do field, range
    z = fill(NaN, size(field.mask))
    z[field.mask] = @. (field.y - G[range]) / sqrt(field.observation_error^2 + field.structural_error^2)
    ρ = [neighbor_correlation(z, dimension) for dimension in 1:ndims(z) if size(z, dimension) > 1]
    (; ρ, misfit = mean(abs2, z[field.mask]), redundancy = 1 / prod(independent_fraction, ρ))
end

for (name, field) in pairs(redundancy)
    @info "$name" correlations = field.ρ independent_values = length(targets[name].y) / field.redundancy field.misfit
end

jldsave(joinpath(@__DIR__, "calibration_data", "redundancy.jld2"); redundancy = map(field -> field.redundancy, redundancy))
