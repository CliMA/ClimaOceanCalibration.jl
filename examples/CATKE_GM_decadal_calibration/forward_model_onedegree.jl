# One member of the decadal calibration: ClimaOcean's `OneDegree()` configuration with scaled CATKE and
# Gent-McWilliams coefficients and sea ice without dynamics, started from ECCO, which writes monthly means of
# temperature, salinity and mixed-layer depth.

using ClimaOcean
using ClimaOcean: OMIPConfiguration
using ClimaSeaIce: ConductiveFlux, UniformThicknessDistribution
using ConstructionBase: setproperties
using Dates
using GibbsSeaWater: gsw_ct_from_pt, gsw_sa_from_sp
using NumericalEarth
using NumericalEarth.Diagnostics: MixedLayerDepthField
using NumericalEarth.SeaIces: default_snow_thermodynamics
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.OutputReaders: InMemoryFTS, MAX_FTS_TUPLE_SIZE, time_indices
using Oceananigans.Grids: λnode, φnode, znode
using Oceananigans.Units

import Oceananigans.BoundaryConditions: fill_halo_regions!

include(joinpath(@__DIR__, "..", "CATKE_GM_calibration", "forward_model_orca.jl"))

# Fills the halos of the forcing records in memory one group after another. On a GPU, the concurrent tasks of
# Oceananigans 0.117 fill them before the records are written (CliMA/Oceananigans.jl#6505).
function fill_halo_regions!(series::InMemoryFTS)
    for indices in Iterators.partition(time_indices(series), MAX_FTS_TUPLE_SIZE)
        fill_halo_regions!(Tuple(series[n] for n in indices))
    end

    return nothing
end

"""
    ThermodynamicSeaIce(settings)

The sea ice `settings` of an `OMIPConfiguration`, for sea ice that has no dynamics and is not advected.
"""
struct ThermodynamicSeaIce{S}
    settings :: S
end

function ClimaOcean.omip_sea_ice(grid, ocean, configuration::OMIPConfiguration{<:Any, <:Any, <:ThermodynamicSeaIce})
    sea_ice = configuration.sea_ice.settings
    itd_shape = UniformThicknessDistribution(categories = sea_ice.thickness_categories)

    simulation = sea_ice_simulation(grid, ocean; dynamics = nothing,
                                    internal_heat_flux = ConductiveFlux(; conductivity = sea_ice.conductivity, itd_shape),
                                    snow_thermodynamics = default_snow_thermodynamics(grid; itd_shape))

    set!(simulation.model, sea_ice.initial_condition)

    return simulation
end

"""
    convert_ecco_to_teos10!(T, S)

Convert the potential temperature `T` [ᵒC] and Practical Salinity `S` of ECCO to Conservative Temperature [ᵒC] and
Absolute Salinity [g kg⁻¹] in place.
"""
function convert_ecco_to_teos10!(T, S)
    grid = on_architecture(CPU(), T.grid)
    θ = Array(interior(T))
    Sᴾ = Array(interior(S))
    Θ = similar(θ)
    Sᴬ = similar(Sᴾ)

    for index in CartesianIndices(θ)
        i, j, k = Tuple(index)
        λ = λnode(i, j, k, grid, Center(), Center(), Center())
        φ = φnode(i, j, k, grid, Center(), Center(), Center())
        p = max(0, -znode(i, j, k, grid, Center(), Center(), Center()))
        Sᴬ[index] = gsw_sa_from_sp(Sᴾ[index], p, λ, φ)
        Θ[index] = gsw_ct_from_pt(Sᴬ[index], θ[index])
    end

    set!(T, Θ)
    set!(S, Sᴬ)

    return nothing
end

"""
    run_decadal_member(scalings; directory,
                       start_date = DateTime(1995, 1, 1),
                       spinup_years = 10,
                       calibration_years = 10,
                       forcing_directory = joinpath(homedir(), "JRA55_data"),
                       ecco_directory = joinpath(homedir(), "ECCO_data"),
                       arch = GPU())

Run `OneDegree()` with the closures of `calibrated_catke(scalings)` and `calibrated_gm(scalings)` and sea ice without
dynamics from the ECCO state of `start_date`, for `spinup_years` and then `calibration_years` of 365 days. Monthly
means of temperature and salinity above 2000 m (`monthly_tracers.jld2`) and of the mixed-layer depth
(`monthly_mixed_layer_depth.jld2`) go to `directory`, with a checkpoint every year from which a second call
continues. Throws if the run stops before its end.
"""
function run_decadal_member(scalings; directory,
                            start_date = DateTime(1995, 1, 1),
                            spinup_years = 10,
                            calibration_years = 10,
                            forcing_directory = joinpath(homedir(), "JRA55_data"),
                            ecco_directory = joinpath(homedir(), "ECCO_data"),
                            arch = GPU())

    ecco = (; dataset = ECCO4Monthly(), date = start_date, dir = ecco_directory)

    configuration = OneDegree(; vertical_mixing = calibrated_catke(scalings),
                                eddy_closure = calibrated_gm(scalings),
                                sea_ice_initial_condition = MetadataSet((:sea_ice_thickness, :sea_ice_concentration); ecco...))

    configuration = setproperties(configuration, (; sea_ice = ThermodynamicSeaIce(configuration.sea_ice)))

    stop_time = (spinup_years + calibration_years) * 365days
    end_date = start_date + Year(spinup_years + calibration_years)

    simulation = omip_simulation(configuration; arch, forcing_directory, stop_time, dates = (start_date, end_date),
                                 diagnostics = false)

    ocean = simulation.model.ocean.model
    grid = ocean.grid

    T = CenterField(grid)
    S = CenterField(grid)
    set!(T, Metadatum(:temperature; ecco...))
    set!(S, Metadatum(:salinity; ecco...))
    convert_ecco_to_teos10!(T, S)
    set!(ocean; T, S)

    month = 365days / 12
    Nz = size(grid, 3)
    k = searchsortedlast(Array(znodes(grid, Face())), -2000)

    # The writers serialize against `simulation.model`, a coupled model that carries none of the ocean properties
    writer_kw = (; dir = directory, including = Symbol[], with_halos = false)

    simulation.output_writers[:tracers] = JLD2Writer(ocean, (; T = ocean.tracers.T, S = ocean.tracers.S); writer_kw...,
                                                     schedule = AveragedTimeInterval(month),
                                                     indices = (:, :, k:Nz),
                                                     filename = "monthly_tracers")

    mixed_layer_depth = MixedLayerDepthField(ocean.buoyancy, grid, ocean.tracers)

    simulation.output_writers[:mixed_layer_depth] = JLD2Writer(ocean, (; h = mixed_layer_depth); writer_kw...,
                                                               schedule = AveragedTimeInterval(month),
                                                               filename = "monthly_mixed_layer_depth")

    simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                            schedule = TimeInterval(365days),
                                                            dir = directory,
                                                            prefix = "checkpoint",
                                                            cleanup = true)

    run!(simulation; pickup = true)

    time(simulation) < stop_time && error("The member in $directory stopped at $(prettytime(simulation)).")

    return simulation
end
