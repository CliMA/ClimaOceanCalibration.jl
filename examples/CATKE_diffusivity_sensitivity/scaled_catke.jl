# CATKE with scaled diffusivities, and the mixed-layer depth of de Boyer Montégut et al. (2004) as a kernel function.

using ConstructionBase: setproperties
using Oceananigans.Grids: znode, static_column_depthᶜᶜᵃ, inactive_cell
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using SeawaterPolynomials: ρ′

"""
    scale_diffusivities(catke, scale)

`catke` with its viscosity and its tracer and turbulent kinetic energy diffusivities multiplied by `scale`: every
mixing length is linear in the stability-function, convective and entrainment coefficients, and the maxima scale
with them. The dissipation length scale is unchanged.
"""
function scale_diffusivities(catke, scale)
    mixing_length = catke.mixing_length
    unscaled = (:Cˢ, :Cᵇ, :Cˢᵖ, :CRiᵟ, :CRi⁰)
    coefficients = filter(∉(unscaled), propertynames(mixing_length))
    scaled = (; (c => scale * getproperty(mixing_length, c) for c in coefficients)...)

    return CATKEVerticalDiffusivity(Oceananigans.TimeSteppers.time_discretization(catke);
                                    mixing_length = setproperties(mixing_length, scaled),
                                    catke.turbulent_kinetic_energy_equation,
                                    maximum_tracer_diffusivity = scale * catke.maximum_tracer_diffusivity,
                                    maximum_tke_diffusivity = scale * catke.maximum_tke_diffusivity,
                                    maximum_viscosity = scale * catke.maximum_viscosity,
                                    catke.minimum_tke,
                                    catke.minimum_convective_buoyancy_flux,
                                    catke.negative_tke_damping_time_scale,
                                    catke.tke_time_step)
end

const c = Center()
const f = Face()

"""The buoyancy that the water of each cell has at the surface, which is the same at every depth for one water mass."""
@inline function potential_buoyancyᶜᶜᶜ(i, j, k, grid, formulation, T, S)
    equation_of_state = formulation.equation_of_state
    ρ′₀ = @inbounds ρ′(T[i, j, k], S[i, j, k], zero(grid), equation_of_state)
    return - formulation.gravitational_acceleration * ρ′₀ / equation_of_state.reference_density
end

"""
The depth below the free surface at which the potential buoyancy `b` has dropped by `Δb★` from its value at the
depth `hʳ`, interpolated between the two levels on either side of the threshold as de Boyer Montégut et al. (2004) do.
"""
@inline function mixed_layer_depthᶜᶜᵃ(i, j, _, grid, b, Δb★, hʳ)
    Nz = size(grid, 3)
    η = znode(i, j, Nz + 1, grid, c, c, f)
    zʳ = η - hʳ

    k⁺ = Nz
    for k in Nz:-1:1
        k⁺ = ifelse(znode(i, j, k, grid, c, c, c) ≥ zʳ, k, k⁺)
    end
    k⁻ = max(k⁺ - 1, 1)
    z⁺ = znode(i, j, k⁺, grid, c, c, c)
    z⁻ = znode(i, j, k⁻, grid, c, c, c)
    weight = clamp((zʳ - z⁻) / max(z⁺ - z⁻, eps(typeof(zʳ))), 0, 1)
    bʳ = @inbounds b[i, j, k⁻] + weight * (b[i, j, k⁺] - b[i, j, k⁻])

    h = static_column_depthᶜᶜᵃ(i, j, grid)
    z₀ = zʳ
    Δb₀ = zero(bʳ)
    mixed = true

    for k in k⁻:-1:1
        zₖ = znode(i, j, k, grid, c, c, c)
        Δb = bʳ - @inbounds b[i, j, k]
        wet = !inactive_cell(i, j, k, grid)
        crossing = mixed & wet & (Δb ≥ Δb★)
        h = ifelse(crossing, η - (z₀ + (zₖ - z₀) * (Δb★ - Δb₀) / (Δb - Δb₀)), h)
        mixed = mixed & wet & !crossing
        z₀ = ifelse(mixed, zₖ, z₀)
        Δb₀ = ifelse(mixed, Δb, Δb₀)
    end

    return h
end
