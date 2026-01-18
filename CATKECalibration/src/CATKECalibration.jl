"""
    CATKECalibration

A module for calibrating CATKE (Convective Adjustment TKE) parameters in ClimaOcean
using Ensemble Kalman Inversion (EKI) with the ClimaCalibrate framework.

This module provides:
- Forward model for running CATKE simulations with scaled parameters
- Data processing utilities for comparing model output to ECCO observations
- BatchedSlurmGCPBackend for efficient multi-GPU job submission on GCP
- ClimaCalibrate interface (forward_model, observation_map, analyze_iteration)

# Usage

```julia
using CATKECalibration

# The module exports key functions for calibration:
# - run_CATKE_calibration_omip: Run forward model with CATKE parameters
# - process_member_data: Process model output for EKI
# - process_monthly_observations: Process ECCO observations
# - build_observation_covariance: Build covariance from multi-year ECCO data
# - BatchedSlurmGCPBackend: HPC backend for batched GPU jobs
```

# Precompilation

The module uses PrecompileTools to precompile the entire forward model run
at full resolution. This significantly reduces startup time for calibration
runs by precompiling all methods used in the simulation.
"""
module CATKECalibration

using PrecompileTools

# Core dependencies
using ClimaOcean
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Models: buoyancy_field, buoyancy_frequency
using Oceananigans.Grids: znodes, φnodes
using Oceananigans.Fields: location, Field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, AdvectiveFormulation, IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.Operators: Δx, Δy

using ClimaOcean.DataWrangling

# Calibration dependencies
using ClimaCalibrate
using ClimaCalibrate: HPCBackend, path_to_iteration, path_to_ensemble_member,
                      path_to_model_log, write_model_started, write_model_completed,
                      model_completed, model_started, wait_for_jobs,
                      generate_sbatch_directives, submit_slurm_job
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses: EnsembleKalmanProcess, get_ϕ, get_ϕ_mean_final, get_error
using EnsembleKalmanProcesses: tsvd_cov_from_samples, SVDplusD

# Data processing dependencies
using XESMF
using JLD2
using NaNStatistics
using Glob
using Statistics
using LinearAlgebra
using TOML

# Other dependencies
using Printf
using Dates
using CUDA
using Random
using Libdl

# Include submodules
include("data_processing.jl")
include("forward_model.jl")
include("slurm_backend.jl")

# Export data processing functions
export compute_dz_weights, regrid_model_data,
       extract_field_section, extract_midlatitude_section,
       process_observation, process_monthly_observations,
       process_member_data, build_observation_covariance

# Export forward model function
export run_CATKE_calibration_omip

# Export Slurm backend
export BatchedSlurmGCPBackend, GPUS_PER_NODE

# Precompile workload for reduced startup time
# This runs a short version of the full forward model at production resolution
# to precompile all methods including output writing
@setup_workload begin
    # Scaling factors for CATKE parameters (default values)
    Cˢ_scaling = 1.0
    Cᵘⁿ_scaling = 1.0
    Cᶜ_scaling = 1.0
    Cˢᵖ_scaling = 1.0
    Cᵉc_scaling = 1.0

    @compile_workload begin
        #####
        ##### CATKE closure setup
        #####
        CATKE_default = ClimaOcean.Oceans.default_ocean_closure()

        Cˢ = CATKE_default.mixing_length.Cˢ * Cˢ_scaling

        Cᵘⁿu = CATKE_default.mixing_length.Cᵘⁿu * Cᵘⁿ_scaling
        Cᵘⁿc = CATKE_default.mixing_length.Cᵘⁿc * Cᵘⁿ_scaling
        Cᵘⁿe = CATKE_default.mixing_length.Cᵘⁿe * Cᵘⁿ_scaling
        CᵘⁿD = CATKE_default.turbulent_kinetic_energy_equation.CᵘⁿD * Cᵘⁿ_scaling

        Cᶜu = CATKE_default.mixing_length.Cᶜu * Cᶜ_scaling
        Cᶜc = CATKE_default.mixing_length.Cᶜc * Cᶜ_scaling
        Cᶜe = CATKE_default.mixing_length.Cᶜe * Cᶜ_scaling
        CᶜD = CATKE_default.turbulent_kinetic_energy_equation.CᶜD * Cᶜ_scaling

        Cᵉc = CATKE_default.mixing_length.Cᵉc * Cᵉc_scaling
        Cˢᵖ = CATKE_default.mixing_length.Cˢᵖ * Cˢᵖ_scaling

        mixing_length = CATKEMixingLength(; Cˢ, Cᵘⁿu, Cᵘⁿc, Cᵘⁿe, Cᶜu, Cᶜc, Cᶜe, Cᵉc, Cˢᵖ, Cᵇ=0.01)
        turbulent_kinetic_energy_equation = CATKEEquation(; Cᵂϵ=1.0, CᵘⁿD, CᶜD)
        catke_closure = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(); mixing_length, turbulent_kinetic_energy_equation)

        #####
        ##### Grid setup - same resolution as production forward model
        #####
        arch = GPU()

        # Full production resolution
        Nx = 720  # longitudinal direction
        Ny = 360  # meridional direction
        Nz = 100

        @info "Building precompilation grid with size ($Nx, $Ny, $Nz)..."

        z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
        z_surf = z_faces(Nz)

        grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))

        bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

        @info "Grid built successfully"

        #####
        ##### Ocean model configuration
        #####
        tracer_advection   = WENO(order=7)
        momentum_advection = WENOVectorInvariant(order=5)
        free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=40minutes)

        # Horizontal viscosity
        @inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) = 2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
        @inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ
        horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters=25days)

        closure = (catke_closure, horizontal_viscosity)

        @info "Building ocean simulation..."

        ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection,
                                tracer_advection,
                                timestepper = :SplitRungeKutta3,
                                free_surface,
                                closure)

        @info "Ocean simulation built"

        #####
        ##### Sea ice model
        #####
        sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
        @info "Sea ice simulation built"

        #####
        ##### Atmosphere and radiation
        #####
        start_year = 2002
        start_date = DateTime(start_year, 1, 1)
        end_date = start_date + Day(3)  # Just 3 days for precompilation

        jra55_dir = joinpath(homedir(), "JRA55_data")
        mkpath(jra55_dir)

        @info "Setting up atmosphere..."
        dataset = MultiYearJRA55()
        backend = JRA55NetCDFBackend(10)

        atmosphere = JRA55PrescribedAtmosphere(arch; dir=jra55_dir, dataset, backend,
                                                include_rivers_and_icebergs=false,
                                                start_date, end_date)
        radiation = Radiation()

        @info "Atmosphere built"

        #####
        ##### Coupled model
        #####
        omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
        @info "Coupled model built"

        #####
        ##### Simulation with output writers
        #####
        simulation_period = Dates.value(Second(end_date - start_date))
        simulation = Simulation(omip, Δt=30minutes, stop_time=simulation_period)

        # Create temporary output directory for precompilation
        precompile_output_dir = mktempdir()

        # Setup output fields (same as production)
        u, v, w = ocean.model.velocities.u, ocean.model.velocities.v, ocean.model.velocities.w
        T, S = ocean.model.tracers.T, ocean.model.tracers.S
        b = buoyancy_field(ocean.model)
        N² = Field(buoyancy_frequency(ocean.model))
        h, ℵ = sea_ice.model.ice_thickness, sea_ice.model.ice_concentration

        ocean_outputs = (; u, v, w, T, S, b, N²)
        sea_ice_outputs = (; h, ℵ)

        # Add output writers to precompile JLD2 writing
        ocean.output_writers[:precompile_ocean] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = joinpath(precompile_output_dir, "ocean_precompile"),
                                            overwrite_existing = true)

        sea_ice.output_writers[:precompile_sea_ice] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = joinpath(precompile_output_dir, "sea_ice_precompile"),
                                            overwrite_existing = true)

        # Simple progress callback
        wall_time = Ref(time_ns())

        function progress(sim)
            step_time = 1e-9 * (time_ns() - wall_time[])
            @info @sprintf("time: %s, iteration: %d, Δt: %s, wall time: %s",
                        prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt), prettytime(step_time))
            wall_time[] = time_ns()
            return nothing
        end

        add_callback!(simulation, progress, IterationInterval(10))

        @info "Simulation ready to run for $(prettytime(simulation_period))"

        #####
        ##### Run simulation for precompilation
        #####
        @info "Starting precompilation simulation..."
        run!(simulation)
        @info "Precompilation simulation completed"

        # Clean up temporary files
        rm(precompile_output_dir, recursive=true, force=true)
    end
end

end # module
