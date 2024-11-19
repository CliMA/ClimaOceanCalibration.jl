module ClimaOceanCalibration

using ClimaOcean
using ClimaOcean.ECCO: ECCO4Monthly
using OrthogonalSphericalShellGrids
using Oceananigans:
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.Units
using CFTime
using Dates
using Printf

include("progress.jl")

# Polar restoring with limits hard coded for now.
@inline function restoring_mask(λ, φ, z, t=0)
    ϵN = (φ - 75) / 5
    ϵN = clamp(ϵN, zero(ϵN), one(ϵN))
    ϵS = - (φ + 75) / 5
    ϵS = clamp(ϵS, zero(ϵS), one(ϵS))
    return ϵN + ϵS
end

@inline sponge_layer(λ, φ, z, t, c, ω) = - restoring_mask(λ, φ, z, t) * ω * c

function diffusive_ocean_simulation(arch=CPU(), FT=Float64;
                                    size = (120, 60, 30),
                                    latitude = (0, 360),
                                    longitude = (-80, 80),
                                    progress_interval = 1,
                                    κ_skew = 1000,
                                    κ_symmetric = 1000)

    Nx, Ny, Nz = size
    z = exponential_z_faces(; Nz, depth=6000)

    #=
    grid = TripolarGrid(arch, FT; z,
                        size = (Nx, Ny, Nz),
                        north_poles_latitude=55,
                        first_pole_longitude=70)
    =#

    grid = LatitudeLongitudeGrid(arch, FT; size, latitude, longitude, z)

    bottom_height = regrid_bathymetry(grid;
                                      minimum_depth = 10,
                                      interpolation_passes = 5,
                                      major_basins = 3)

    # Closure
    gm = IsopycnalSkewSymmetricDiffusivity(FT; κ_skew, κ_symmetric)
    catke = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(FT)
    closure = (gm, catke)

    restoring_rate = 1 / 1days
    restoring_mask_field = CenterField(grid)
    set!(restoring_mask_field, restoring_mask)

    Fu = Forcing(sponge_layer, field_dependencies=:u, parameters=restoring_rate)
    Fv = Forcing(sponge_layer, field_dependencies=:v, parameters=restoring_rate)

    dates = DateTimeProlepticGregorian(1993, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 1, 1)
    temperature = ECCOMetadata(:temperature, dates, ECCO4Monthly())
    salinity = ECCOMetadata(:salinity, dates, ECCO4Monthly())

    FT = ECCORestoring(arch, temperature; grid, mask=restoring_mask_field, rate=restoring_rate)
    FS = ECCORestoring(arch, salinity;    grid, mask=restoring_mask_field, rate=restoring_rate)
    forcing = (T=FT, S=FS, u=Fu, v=Fv)

    ocean = ocean_simulation(grid; closure, forcing,
                             momentum_advection = VectorInvariant(),
                             tracer_advection = Centered(order=2),
                             tracers = (:T, :S, :e))

    set!(ocean.model,
         T = ECCOMetadata(:temperature; dates=first(dates)),
         S = ECCOMetadata(:salinity; dates=first(dates)))

    radiation = Radiation(arch)
    atmosphere = JRA55_prescribed_atmosphere(arch; backend=JRA55NetCDFBackend(41))
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
    simulation = Simulation(coupled_model; Δt=10minutes)
    add_callback!(simulation, Progress(), IterationInterval(progress_interval))

    return simulation
end

function reschedule!(simulation, name, new_schedule)
    cb = simulation.callbacks[name]
    new_cb = Callback(cb.func, new_schedule; parameters=cb.parameters, callsite=cb.callsite)
    simulation.callbacks[name] = new_cb
    return nothing
end

end # module ClimaOceanCalibration
