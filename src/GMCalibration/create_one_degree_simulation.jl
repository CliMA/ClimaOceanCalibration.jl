using ClimaOcean
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids
using ClimaOcean.OceanSimulations
using ClimaOcean.ECCO
using ClimaOcean.JRA55
using ClimaOcean.DataWrangling
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using Printf
using Dates
using CUDA
using Oceananigans.TurbulenceClosures: DiffusiveFormulation, AdvectiveFormulation
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity
using Oceananigans.BuoyancyFormulations: buoyancy, buoyancy_frequency

import Oceananigans.OutputWriters: checkpointer_address

function create_one_degree_omip(; arch = Oceananigans.GPU(),
                                  κ_skew = 1.5e3,
                                  κ_symmetric = 1.5e3,
                                  skew_flux_formulation = AdvectiveFormulation(),
                                  vertical_mixing_closure = RiBasedVerticalDiffusivity(),
                                  momentum_advection = WENOVectorInvariant(order = 5),
                                  tracer_advection = WENO(order = 5),
                                  forcing_dir = "./OMIP_forcing_data",
                                  output_dir = nothing,
                                  output_interval = 365days,
                                  checkpoint_interval = 3650days,
                                  simulation_name = nothing,
                                  sea_ice_dynamics = nothing)
                                 
    # Grid parameters
    depth = 6000
    Nx = 360
    Ny = 180
    Nz = 60
    
    # Set up grid
    z_faces = ExponentialCoordinate(Nz, -depth, 0)
    z_surf = z_faces(Nz)
    
    grid = TripolarGrid(arch;
                        size = (Nx, Ny, Nz),
                        z = z_faces,
                        halo = (7, 7, 7))
    
    bottom_height = regrid_bathymetry(grid; minimum_depth = 15, major_basins = 1, interpolation_passes = 75)
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)
    
    # Set up free surface
    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.8, fixed_Δt = 65minutes)
    
    # Configure GM eddy closure
    eddy_closure = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(
        κ_skew = κ_skew, 
        κ_symmetric = κ_symmetric, 
        skew_flux_formulation = skew_flux_formulation
    )
    
    # Build closure
    closure = (vertical_mixing_closure, VerticalScalarDiffusivity(κ = 1e-5, ν = 1e-4), eddy_closure)
    
    # Generate simulation name if not provided
    if isnothing(simulation_name)
        simulation_name = "one_degree_OMIP_skew_$(κ_skew)_symmetric_$(κ_symmetric)"
    end
    
    # Configure output directory
    if isnothing(output_dir)
        FILE_DIR = "./calibration_runs/$(simulation_name)/"
    else
        FILE_DIR = "$(output_dir)/$(simulation_name)/"
    end
    mkpath(FILE_DIR)

    if isnothing(forcing_dir)
        forcing_dir = "./OMIP_forcing_data"
    end
    mkpath(forcing_dir)
    
    # Configure salinity restoring
    @inline mask(x, y, z, t) = z ≥ z_surf - 1
    dataset = EN4Monthly()
    date = DateTime(1958, 1, 1)
    Smetadata = Metadata(:salinity; dataset, dir = forcing_dir)
    
    FS = DatasetRestoring(Smetadata, grid; rate = 1/18days, mask, time_indices_in_memory = 10)
    
    # Set up ocean simulation
    ocean = ocean_simulation(grid; 
                             Δt = 1minutes,
                             momentum_advection = momentum_advection,
                             tracer_advection = tracer_advection,
                             timestepper = :SplitRungeKutta3,
                             free_surface,
                             forcing = (; S = FS),
                             closure)
    
    # Set initial conditions
    set!(ocean.model, 
         T = Metadatum(:temperature; dataset, date, forcing_dir),
         S = Metadatum(:salinity; dataset, date, forcing_dir))
    
    # Set up sea ice model
    sea_ice = sea_ice_simulation(grid, ocean; dynamics = sea_ice_dynamics)
    
    set!(sea_ice.model, 
         h = Metadatum(:sea_ice_thickness; dataset = ECCO4Monthly(), dir = forcing_dir),
         ℵ = Metadatum(:sea_ice_concentration; dataset = ECCO4Monthly(), dir = forcing_dir))
    
    # Set up atmosphere model
    dataset_atmosphere = MultiYearJRA55()
    backend = JRA55NetCDFBackend(100)
    
    atmosphere = JRA55PrescribedAtmosphere(arch; 
                                          dir = forcing_dir, 
                                          dataset = dataset_atmosphere, 
                                          backend, 
                                          include_rivers_and_icebergs = true)
    radiation = Radiation()
    
    # Create coupled model
    omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    simulation = Simulation(omip, Δt = 20minutes, stop_time = 60days)
    
    # Set up output writers
    setup_output_writers!(ocean, sea_ice, grid, FILE_DIR, output_interval, checkpoint_interval)
    
    # Set up progress callback
    add_progress_callback!(simulation)
    
    return simulation
end

# Helper function to set up output writers
function setup_output_writers!(ocean, sea_ice, grid, FILE_DIR, output_interval, checkpoint_interval)
    # Define checkpointer address function
    checkpointer_address(::SeaIceModel) = "SeaIceModel"
    
    # Set up checkpointers
    ocean.output_writers[:checkpointer] = Checkpointer(ocean.model,
                                                      schedule = TimeInterval(checkpoint_interval),
                                                      prefix = "$(FILE_DIR)/ocean_checkpoint",
                                                      overwrite_existing = true)
    
    sea_ice.output_writers[:checkpointer] = Checkpointer(sea_ice.model,
                                                       schedule = TimeInterval(checkpoint_interval),
                                                       prefix = "$(FILE_DIR)/sea_ice_checkpoint",
                                                       overwrite_existing = true)
                                                     
    # Set up field outputs
    u, v, w = ocean.model.velocities
    T, S = ocean.model.tracers
    b = Field(buoyancy(ocean.model))
    N² = Field(buoyancy_frequency(ocean.model))
    
    ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities, (; b, N²))
    
    # Zonal average outputs
    ubar = Average(u, dims = 1)
    vbar = Average(v, dims = 1)
    wbar = Average(w, dims = 1)
    Tbar = Average(T, dims = 1)
    Sbar = Average(S, dims = 1)
    bbar = Average(b, dims = 1)
    N²bar = Average(N², dims = 1)
    
    ocean_zonal_average_outputs = (; ubar, vbar, wbar, Tbar, Sbar, bbar, N²bar)
    
    # Sea ice outputs
    sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                           ℵ = sea_ice.model.ice_concentration,
                           T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
                           sea_ice.model.velocities)
    
    # Surface outputs
    ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                              schedule = TimeInterval(output_interval),
                                              filename = "$(FILE_DIR)/ocean_surface_fields",
                                              indices = (:, :, grid.Nz),
                                              overwrite_existing = true)
    
    sea_ice.output_writers[:surface] = JLD2Writer(ocean.model, sea_ice_outputs;
                                                schedule = TimeInterval(output_interval),
                                                filename = "$(FILE_DIR)/sea_ice_surface_fields",
                                                overwrite_existing = true)
    
    # Full 3D outputs
    ocean.output_writers[:full] = JLD2Writer(ocean.model, ocean_outputs;
                                           schedule = TimeInterval(checkpoint_interval),
                                           filename = "$(FILE_DIR)/ocean_complete_fields",
                                           overwrite_existing = true)
    
    # Time averaged outputs
    ocean.output_writers[:time_average] = JLD2Writer(ocean.model, ocean_outputs;
                                                   schedule = AveragedTimeInterval(checkpoint_interval, window = checkpoint_interval),
                                                   filename = "$(FILE_DIR)/ocean_complete_fields_average",
                                                   overwrite_existing = true)
    
    sea_ice.output_writers[:time_average] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                     schedule = AveragedTimeInterval(checkpoint_interval, window = checkpoint_interval),
                                                     filename = "$(FILE_DIR)/sea_ice_complete_fields_average",
                                                     overwrite_existing = true)
    
    return nothing
end

# Helper function to add progress callback
function add_progress_callback!(simulation)
    wall_time = Ref(time_ns())
    
    function progress(sim)
        sea_ice = sim.model.sea_ice
        ocean   = sim.model.ocean
        hmax = maximum(sea_ice.model.ice_thickness)
        ℵmax = maximum(sea_ice.model.ice_concentration)
        Tmax = maximum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        Tmin = minimum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        umax = maximum(ocean.model.velocities.u)
        vmax = maximum(ocean.model.velocities.v)
        wmax = maximum(ocean.model.velocities.w)
        
        step_time = 1e-9 * (time_ns() - wall_time[])
        
        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
        msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
        msg4 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
        msg5 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m/s, ", umax, vmax, wmax)
        msg6 = @sprintf("wall time: %s \n", prettytime(step_time))
        
        @info msg1 * msg2 * msg4 * msg5 * msg6
        
        wall_time[] = time_ns()
        
        return nothing
    end
    
    add_callback!(simulation, progress, IterationInterval(100))
    return nothing
end
