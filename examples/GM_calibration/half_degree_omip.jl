using ClimaOcean
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.BuoyancyFormulations: buoyancy, buoyancy_frequency
using ClimaOcean.OceanSimulations
using ClimaOcean.ECCO
using ClimaOcean.JRA55
using ClimaOcean.DataWrangling
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using Printf
using Dates
using CUDA
using JLD2
using ArgParse
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, AdvectiveFormulation, IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.Operators: Δx, Δy
using Statistics

import Oceananigans.OutputWriters: checkpointer_address

using Libdl
ucx_libs = filter(lib -> occursin("ucx", lowercase(lib)), Libdl.dllist())
if isempty(ucx_libs)
    @info "✓ No UCX - safe to run!"
else
    @warn "✗ UCX libraries detected! This can cause issues with MPI+CUDA. Detected libs:\n$(join(ucx_libs, "\n"))"
end

function run_gm_calibration_omip(κ_skew, κ_symmetric, config_dict)
    output_dir = config_dict["output_dir"]
    logfile_path = joinpath(output_dir, "output.log")

    logfile = open(logfile_path, "w")
    original_stdout = stdout
    original_stderr = stderr
    
    redirect_stdout(logfile)
    redirect_stderr(logfile)

    flusher = @async while isopen(logfile); flush(logfile); sleep(1); end
    
    try
        start_year = 1992
        simulation_length = config_dict["simulation_length"]
        sampling_length = config_dict["sampling_length"]

        @info "Using κ_skew = $(κ_skew) m²/s and κ_symmetric = $(κ_symmetric) m²/s, starting in $(start_year) for $(simulation_length) years with $(sampling_length)-year sampling window."
        @info "Saving output to $(config_dict["output_dir"])"

        arch = GPU()

        Nx = 720 # longitudinal direction 
        Ny = 360 # meridional direction 
        Nz = 100

        z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
        z_surf = z_faces(Nz)

        grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))

        bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

        momentum_advection = WENOVectorInvariant(order=5)
        tracer_advection   = WENO(order=7)
        free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=50minutes)

        @inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
        @inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

        eddy_closure  = IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric, skew_flux_formulation=AdvectiveFormulation())
        # obl_closure = ClimaOcean.OceanSimulations.default_ocean_closure()
        obl_closure = RiBasedVerticalDiffusivity()
        visc_closure  = HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters=25days)

        closure = (obl_closure, VerticalScalarDiffusivity(κ=1e-5, ν=3e-4), visc_closure, eddy_closure)

        prefix = "halfdegree"
        if obl_closure isa RiBasedVerticalDiffusivity
            prefix *= "_RiBased"
        else
            prefix *= "_CATKE"
        end

        prefix *= "_$(κ_skew)_$(κ_symmetric)"
        prefix *= "_$(start_year)"
        prefix *= "_$(simulation_length)year"
        prefix *= "_advectiveGM_multiyearjra55_calibrationsamples"

        dir = joinpath(homedir(), "forcing_data_half_degree")
        mkpath(dir)

        start_date = DateTime(start_year, 1, 1)
        end_date = start_date + Year(simulation_length)
        simulation_period = Dates.value(Second(end_date - start_date))
        sampling_start_date = end_date - Year(sampling_length)
        sampling_window = Dates.value(Second(end_date - sampling_start_date))

        @info "Settting up salinity restoring..."
        @inline mask(x, y, z, t) = z ≥ z_surf - 1
        Smetadata = Metadata(:salinity; dataset=EN4Monthly(), dir, start_date, end_date)
        FS = DatasetRestoring(Smetadata, grid; rate = 1/30days, mask, time_indices_in_memory = 10)

        ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection,
                                tracer_advection,
                                timestepper = :SplitRungeKutta3,
                                free_surface,
                                forcing = (; S = FS),
                                closure)

        @info "Built ocean model $(ocean)"

        set!(ocean.model, T=Metadatum(:temperature; dataset=EN4Monthly(), date=start_date, dir),
                        S=Metadatum(:salinity;    dataset=EN4Monthly(), date=start_date, dir))
        @info "Initialized T and S"

        # Default sea-ice dynamics and salinity coupling are included in the defaults
        # sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))
        sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
        @info "Built sea ice model $(sea_ice)"

        set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir),
                            ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir))

        @info "Initialized sea ice fields"

        jra55_dir = joinpath(homedir(), "JRA55_data")
        mkpath(jra55_dir)
        dataset = MultiYearJRA55()
        backend = JRA55NetCDFBackend(100)

        @info "Setting up presctibed atmosphere $(dataset)"
        atmosphere = JRA55PrescribedAtmosphere(arch; dir=jra55_dir, dataset, backend, include_rivers_and_icebergs=true, start_date, end_date)
        radiation  = Radiation()

        @info "Built atmosphere model $(atmosphere)"

        omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

        @info "Built coupled model $(omip)"

        omip = Simulation(omip, Δt=30minutes, stop_time=simulation_period) 
        @info "Built simulation $(omip)"

        FILE_DIR = config_dict["output_dir"]
        mkpath(FILE_DIR)

        b = Field(buoyancy(ocean.model))
        N² = Field(buoyancy_frequency(ocean.model))

        ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities, (; b, N²))
        sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                                ℵ = sea_ice.model.ice_concentration,
                                T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
                                sea_ice.model.velocities)

        ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                                    schedule = TimeInterval(180days),
                                                    filename = "$(FILE_DIR)/ocean_surface_fields",
                                                    indices = (:, :, grid.Nz),
                                                    overwrite_existing = true)

        sea_ice.output_writers[:surface] = JLD2Writer(ocean.model, sea_ice_outputs;
                                                    schedule = TimeInterval(180days),
                                                    filename = "$(FILE_DIR)/sea_ice_surface_fields",
                                                    overwrite_existing = true)

        ocean.output_writers[:time_average] = JLD2Writer(ocean.model, ocean_outputs;
                                                        schedule = AveragedTimeInterval(3650days, window=3650days),
                                                        filename = "$(FILE_DIR)/ocean_complete_fields_10year_average",
                                                        overwrite_existing = true)

        sea_ice.output_writers[:time_average] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                        schedule = AveragedTimeInterval(3650days, window=3650days),
                                                        filename = "$(FILE_DIR)/sea_ice_complete_fields_10year_average",
                                                        overwrite_existing = true)

        ocean.output_writers[:sample_decadal_average] = JLD2Writer(ocean.model, ocean_outputs;
                                                                schedule = AveragedTimeInterval(simulation_period, window=sampling_window),
                                                                filename = "$(FILE_DIR)/ocean_complete_fields_10year_average_calibrationsample",
                                                                overwrite_existing = true)

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

        # And add it as a callback to the simulation.
        add_callback!(omip, progress, IterationInterval(100))

        run!(omip)
        return nothing
    catch e
        # Handle errors
        if e isa InterruptException
            println(stderr, "Interrupted by user")
        else
            println(stderr, "Error occurred: $e")
            # Optionally rethrow to propagate the error
            rethrow(e)
        end
    finally
        # Cleanup - ALWAYS runs
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
        close(logfile)
        println("Log file closed")  # This prints to console, not log
    end

end

function run_gm_calibration_omip_dry_run(κ_skew, κ_symmetric, config_dict)
    output_dir = config_dict["output_dir"]
    logfile_path = joinpath(output_dir, "output.log")

    logfile = open(logfile_path, "w")
    original_stdout = stdout
    original_stderr = stderr
    
    redirect_stdout(logfile)
    redirect_stderr(logfile)

    flusher = @async while isopen(logfile); flush(logfile); sleep(1); end
    
    try
        start_year = rand(1992:2011)
        @info "Dry run: Using κ_skew = $(κ_skew) m²/s and κ_symmetric = $(κ_symmetric) m²/s, starting in year $(start_year)"
        @info "Saving output to $(config_dict["output_dir"])"

        arch = GPU()

        Nx = 720 # longitudinal direction 
        Ny = 360 # meridional direction 
        Nz = 100

        z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
        z_surf = z_faces(Nz)

        grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))

        bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

        momentum_advection = WENOVectorInvariant(order=5)
        tracer_advection   = WENO(order=7)
        free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=50minutes)

        @inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
        @inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

        eddy_closure  = IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric, skew_flux_formulation=AdvectiveFormulation())
        obl_closure = RiBasedVerticalDiffusivity()
        visc_closure  = HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters=25days)

        closure = (obl_closure, VerticalScalarDiffusivity(κ=1e-5, ν=3e-4), visc_closure, eddy_closure)

        dir = joinpath(homedir(), "forcing_data_half_degree")
        mkpath(dir)

        start_date = DateTime(start_year, 1, 1)
        end_date = start_date + Month(2)
        simulation_period = Dates.value(Second(end_date - start_date))

        @info "Settting up salinity restoring..."
        @inline mask(x, y, z, t) = z ≥ z_surf - 1
        Smetadata = Metadata(:salinity; dataset=EN4Monthly(), dir, start_date, end_date)
        FS = DatasetRestoring(Smetadata, grid; rate = 1/30days, mask, time_indices_in_memory = 2)

        ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection,
                                tracer_advection,
                                timestepper = :SplitRungeKutta3,
                                free_surface,
                                forcing = (; S = FS),
                                closure)

        @info "Built ocean model $(ocean)"

        set!(ocean.model, T=Metadatum(:temperature; dataset=EN4Monthly(), date=start_date, dir),
                        S=Metadatum(:salinity;    dataset=EN4Monthly(), date=start_date, dir))
        @info "Initialized T and S"

        sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
        @info "Built sea ice model $(sea_ice)"

        set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir),
                            ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir))

        @info "Initialized sea ice fields"

        jra55_dir = joinpath(homedir(), "JRA55_data")
        mkpath(jra55_dir)
        dataset = MultiYearJRA55()
        backend = JRA55NetCDFBackend(100)

        @info "Setting up presctibed atmosphere $(dataset)"
        atmosphere = JRA55PrescribedAtmosphere(arch; dir=jra55_dir, dataset, backend, include_rivers_and_icebergs=true, start_date, end_date)
        radiation  = Radiation()

        @info "Built atmosphere model $(atmosphere)"

        omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

        @info "Built coupled model $(omip)"

        omip = Simulation(omip, Δt=30minutes, stop_time=simulation_period)
        @info "Built simulation $(omip)"

        FILE_DIR = config_dict["output_dir"]
        mkpath(FILE_DIR)

        b = Field(buoyancy(ocean.model))
        N² = Field(buoyancy_frequency(ocean.model))

        ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities, (; b, N²))

        ocean.output_writers[:sample_decadal_average] = JLD2Writer(ocean.model, ocean_outputs;
                                                                schedule = AveragedTimeInterval(simulation_period, window=30days),
                                                                filename = "$(FILE_DIR)/ocean_complete_fields_10year_average_calibrationsample",
                                                                overwrite_existing = true)

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

            msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
            msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
            msg4 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
            msg5 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m/s, ", umax, vmax, wmax)
            msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

            @info msg1 * msg2 * msg4 * msg5 * msg6

            wall_time[] = time_ns()

            return nothing
        end

        add_callback!(omip, progress, IterationInterval(10))

        run!(omip)
        return nothing
    catch e
        # Handle errors
        if e isa InterruptException
            println(stderr, "Interrupted by user")
        else
            println(stderr, "Error occurred: $e")
            # Optionally rethrow to propagate the error
            rethrow(e)
        end
    finally
        # Cleanup - ALWAYS runs
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
        close(logfile)
        println("Log file closed")  # This prints to console, not log
    end
end

# function run_gm_calibration_omip_dry_run(κ_skew, κ_symmetric, config_dict)
#     output_dir = config_dict["output_dir"]
#     logfile_path = joinpath(output_dir, "output.log")

#     logfile = open(logfile_path, "w")
#     original_stdout = stdout
#     original_stderr = stderr
    
#     redirect_stdout(logfile)
#     redirect_stderr(logfile)

#     flusher = @async while isopen(logfile); flush(logfile); sleep(1); end
    
#     try
#         # ALL your main code goes here
#         println("Starting work...")
        
#         start_year = rand(1992:2011)
#         member = config_dict["member"]
#         iteration = config_dict["iteration"]
#         @info "Member $member, iter $iteration dry run: Using κ_skew = $(κ_skew) m²/s and κ_symmetric = $(κ_symmetric) m²/s, starting in year $(start_year)"
#         @info "Saving output to $(config_dict["output_dir"])"
#         FILE_DIR = config_dict["output_dir"]
#         mkpath(FILE_DIR)

#         cp(joinpath(homedir(), "ocean_complete_fields_10year_average_calibrationsample.jld2"), "$(FILE_DIR)/ocean_complete_fields_10year_average_calibrationsample.jld2")
        
#         println("Finished successfully")
        
#         return nothing
#     catch e
#         # Handle errors
#         if e isa InterruptException
#             println(stderr, "Interrupted by user")
#         else
#             println(stderr, "Error occurred: $e")
#             # Optionally rethrow to propagate the error
#             rethrow(e)
#         end
#     finally
#         # Cleanup - ALWAYS runs
#         redirect_stdout(original_stdout)
#         redirect_stderr(original_stderr)
#         close(logfile)
#         println("Log file closed")  # This prints to console, not log
#     end
# end