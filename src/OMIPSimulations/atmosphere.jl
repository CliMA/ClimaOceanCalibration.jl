"""
    omip_forcing(arch, sea_ice; forcing_dir, start_date, end_date,
                 repeat_year_forcing=false, backend_size=30)

Build the prescribed atmosphere forcing for an OMIP-2 simulation: JRA55-do
atmosphere and JRA55-do downwelling radiation (with OMIP-2 ocean surface
properties and CCSM3 temperature/snow/thickness-dependent sea-ice albedo).

The JRA55-do land freshwater forcing (river runoff + iceberg calving) is built
separately by [`omip_simulation`](@ref) via [`omip_land_forcing`](@ref): as of
NumericalEarth 0.6, `JRA55PrescribedLand` takes a *grid* rather than an
architecture (it needs the grid to route rivers to their ocean outlets), and the
grid is not available here. This mirrors NumericalEarth's own
experiments/OMIPSimulations on `ss/omip-prototype`.

Returns the tuple `(atmosphere, radiation)`.
"""
function omip_forcing(arch, sea_ice;
                      forcing_dir,
                      start_date,
                      end_date,
                      repeat_year_forcing = false,
                      backend_size = 30,
                      prefetch = true)

    dataset = repeat_year_forcing ? RepeatYearJRA55() : MultiYearJRA55()

    kw = (; dir = forcing_dir,
            dataset,
            start_date,
            end_date,
            time_indices_in_memory = backend_size,
            prefetch)

    atmosphere = JRA55PrescribedAtmosphere(arch; kw...)

    # CCSM3 sea-ice albedo reads live model fields, so the surface
    # temperature must come from whichever layer the atmosphere actually
    # sees: snow top if a snow model is present, ice top otherwise.
    hi = sea_ice.model.ice_thickness
    hs = sea_ice.model.snow_thickness
    snow_thermo = sea_ice.model.snow_thermodynamics
    Ts = isnothing(snow_thermo) ? sea_ice.model.ice_thermodynamics.top_surface_temperature :
                                  snow_thermo.top_surface_temperature
    sea_ice_albedo = SeaIceAlbedo(hi, hs, Ts)

    radiation = JRA55PrescribedRadiation(arch;
                                         kw...,
                                         ocean_surface   = SurfaceRadiationProperties(0.06, 1.00),
                                         sea_ice_surface = SurfaceRadiationProperties(sea_ice_albedo, 1.0))

    return atmosphere, radiation
end

"""
    omip_land_forcing(grid; forcing_dir, start_date, end_date,
                      repeat_year_forcing=false, backend_size=30, prefetch=true)

Build the JRA55-do prescribed land forcing (river runoff + iceberg calving) on
`grid`.

NumericalEarth 0.6 changed `JRA55PrescribedLand` to take the *grid* instead of an
architecture: it resolves each river's discharge cell by searching the grid for an
ocean outlet, so it needs the bathymetry. Passing an architecture now fails with
`MethodError: no method matching architecture(::CPU)`.

`maximum_search_radius` follows NumericalEarth's experiments/OMIPSimulations: a
fixed ~3° geographic reach converted to cells, floored at the upstream default of
5, so the routing search is resolution-independent. At ORCA1 this evaluates to
exactly 5 (i.e. no change from the default); it only widens on finer grids.
"""
function omip_land_forcing(grid;
                           forcing_dir,
                           start_date,
                           end_date,
                           repeat_year_forcing = false,
                           backend_size = 30,
                           prefetch = true)

    dataset = repeat_year_forcing ? RepeatYearJRA55() : MultiYearJRA55()

    Nx, Ny, _ = size(grid)
    maximum_search_radius = max(5, ceil(Int, 3 / ((360 / Nx + 180 / Ny) / 2)))

    return JRA55PrescribedLand(grid;
                               dir = forcing_dir,
                               dataset,
                               start_date,
                               end_date,
                               time_indices_in_memory = backend_size,
                               prefetch,
                               maximum_search_radius)
end
