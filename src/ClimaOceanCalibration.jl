module ClimaOceanCalibration

using Oceananigans
using Oceananigans.Simulations: reset!
using ClimaOcean
using ClimaOcean.ECCO: ECCO4Monthly
using OrthogonalSphericalShellGrids
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.Units
using CFTime
using Dates
using Printf

include("progress.jl")

struct NorthSouthMask{FT} <: Function
    southern_limit :: FT
    southern_width :: FT
    northern_limit :: FT
    northern_width :: FT
end

function NorthSouthMask(φS, δS, φN, δN; float_type=Float64)
    FT = float_type
    return NorthSouthMask(convert(FT, φS),
                          convert(FT, δS),
                          convert(FT, φN),
                          convert(FT, δN))
end

# Polar restoring with limits hard coded for now.
@inline function (m::NorthSouthMask)(λ, φ, z, t=0)
    φS = m.southern_limit
    δS = m.southern_width
    φN = m.northern_limit
    δN = m.northern_width

    ϵN = (φ - φN) / δN
    ϵN = clamp(ϵN, zero(ϵN), one(ϵN))
    ϵS = - (φ - φS) / δS
    ϵS = clamp(ϵS, zero(ϵS), one(ϵS))

    return ϵN + ϵS
end

@inline u_sponge(i, j, k, grid, clock, fields, p) = @inbounds - p.μ[i, j, k] * p.ω * fields.u[i, j, k]
@inline v_sponge(i, j, k, grid, clock, fields, p) = @inbounds - p.μ[i, j, k] * p.ω * fields.v[i, j, k]

function default_closure(FT)
    κ_skew = 1000
    κ_symmetric = 1000
    gm = IsopycnalSkewSymmetricDiffusivity(FT; κ_skew, κ_symmetric)
    catke = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(FT)
    return (gm, catke)
end

function diffusive_ocean_simulation(arch=CPU(), FT=Float64;
                                    size = (120, 60, 10),
                                    longitude = (0, 360),
                                    latitude = (-80, 80),
                                    closure = default_closure(FT),
                                    restoring_mask = NorthSouthMask(-75, 5, +75, 5, float_type=FT),
                                    progress_interval = 1)

    Nx, Ny, Nz = size
    z = exponential_z_faces(; Nz, depth=6000)

    grid = LatitudeLongitudeGrid(arch, FT; size, latitude, longitude, z)

    bottom_height = regrid_bathymetry(grid;
                                      minimum_depth = 10,
                                      interpolation_passes = 1,
                                      major_basins = 1)

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

    # Closure
    restoring_rate = 1 / 1days
    restoring_mask_field = CenterField(grid)
    set!(restoring_mask_field, restoring_mask)

    Fu = Forcing(u_sponge, discrete_form=true, parameters=(ω=restoring_rate, μ=restoring_mask_field))
    Fv = Forcing(v_sponge, discrete_form=true, parameters=(ω=restoring_rate, μ=restoring_mask_field))

    dates = DateTimeProlepticGregorian(1993, 1, 1) : Month(1) : DateTimeProlepticGregorian(1994, 1, 1)
    temperature = ECCOMetadata(:temperature, dates, ECCO4Monthly())
    salinity = ECCOMetadata(:salinity, dates, ECCO4Monthly())

    FT = ECCORestoring(temperature, grid; mask=restoring_mask_field, rate=restoring_rate)
    FS = ECCORestoring(salinity, grid;    mask=restoring_mask_field, rate=restoring_rate)
    forcing = (T=FT, S=FS, u=Fu, v=Fv)

    ocean = ocean_simulation(grid; closure, forcing,
                             momentum_advection = VectorInvariant(),
                             tracer_advection = Centered(order=2),
                             tracers = (:T, :S, :e))

    radiation = Radiation(arch)
    atmosphere = JRA55_prescribed_atmosphere(arch; backend=JRA55NetCDFBackend(41))
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
    simulation = Simulation(coupled_model; Δt=10minutes, verbose=false)
    add_callback!(simulation, Progress(), IterationInterval(progress_interval))
    reset_coupled_simulation!(simulation)

    return simulation
end

function reschedule!(simulation, name, new_schedule)
    cb = simulation.callbacks[name]
    new_cb = Callback(cb.func, new_schedule; parameters=cb.parameters, callsite=cb.callsite)
    simulation.callbacks[name] = new_cb
    return nothing
end

function reset_coupled_simulation!(simulation)
    ocean = simulation.model.ocean
    reset!(simulation)
    reset!(ocean)
    
    @show simulation
    @show ocean

    dates = DateTimeProlepticGregorian(1993, 1, 1)

    set!(ocean.model,
         T = ECCOMetadata(:temperature; dates),
         S = ECCOMetadata(:salinity; dates))

    return nothing
end

end # module ClimaOceanCalibration

