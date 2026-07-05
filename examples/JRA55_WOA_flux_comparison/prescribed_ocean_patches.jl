# prescribed_ocean_patches.jl
#
# Two small method extensions on NumericalEarth 0.5.7's `PrescribedOcean` that
# enable the full net-flux + radiation pipeline for a prescribed (non-dynamic)
# ocean. Same in-script-patch pattern as src/OMIPSimulations/oceananigans_patches.jl
# and NumericalEarth's own experiments/flux_climatology/flux_climatology.jl.
# Both are upstream-PR candidates.
#
# 1. `net_fluxes(::PrescribedOcean)` upstream returns `nothing`, which
#    (a) skips net-flux assembly (no hfds/wfo-equivalent diagnostics), and
#    (b) breaks `apply_air_sea_radiative_fluxes!`, which passes
#        `interfaces.net_fluxes.ocean` into its kernel unguarded.
#    Here we allocate the four standard net-flux fields (same locations as the
#    ocean_simulation boundary-condition fluxes) so both kernels have a target.
#
# 2. `update_net_fluxes!(coupled_model, ::PrescribedOcean)` upstream is a no-op.
#    We route it to the standard `update_net_ocean_fluxes!` assembly, which is
#    generic in the ocean component (it only needs `ocean_surface_salinity`,
#    already defined for `PrescribedOcean`). With no sea-ice component,
#    `computed_fluxes(nothing) = ZeroFluxes()` and
#    `sea_ice_concentration(nothing) = ZeroField()` make it an open-water assembly.
#
# IMPORTANT: include this file BEFORE constructing `ComponentInterfaces`
# (which calls `net_fluxes(ocean)` once, at construction).

using Oceananigans
using Oceananigans.Fields: Field, Center, Face
using NumericalEarth
using NumericalEarth.Oceans: PrescribedOcean, update_net_ocean_fluxes!
using NumericalEarth.EarthSystemModels: EarthSystemModels
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations

# Allocate once per ocean instance (ComponentInterfaces calls this exactly once,
# but memoize anyway so repeated construction in a REPL reuses the same fields).
const _PRESCRIBED_NET_FLUXES = IdDict{Any, Any}()

function InterfaceComputations.net_fluxes(ocean::PrescribedOcean)
    get!(_PRESCRIBED_NET_FLUXES, ocean) do
        grid = ocean.grid
        τx = Field{Face, Center, Nothing}(grid)     # kinematic x-stress [m² s⁻²]
        τy = Field{Center, Face, Nothing}(grid)     # kinematic y-stress [m² s⁻²]
        JT = Field{Center, Center, Nothing}(grid)   # net temperature flux [K m s⁻¹] (turbulent + radiative)
        JS = Field{Center, Center, Nothing}(grid)   # net salinity flux [psu m s⁻¹]
        (; u = τx, v = τy, T = JT, S = JS)
    end
end

EarthSystemModels.update_net_fluxes!(coupled_model, ocean::PrescribedOcean) =
    update_net_ocean_fluxes!(coupled_model, ocean, ocean.grid)

@info "PrescribedOcean patches loaded: net_fluxes allocation + net-flux assembly enabled."
