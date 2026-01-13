using ClimaOcean
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Models: buoyancy_field, buoyancy_frequency
using ClimaOcean.DataWrangling
using Printf
using Dates
using CUDA
using JLD2
using ArgParse
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, AdvectiveFormulation, IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.Operators: Δx, Δy
using Oceananigans: prognostic_fields
using Oceananigans.AbstractOperations: condition_operand
using Statistics
using EnsembleKalmanProcesses
using Random

import Oceananigans.OutputWriters: checkpointer_address

using Libdl
ucx_libs = filter(lib -> occursin("ucx", lowercase(lib)), Libdl.dllist())
if isempty(ucx_libs)
    @info "✓ No UCX - safe to run!"
else
    @warn "✗ UCX libraries detected! This can cause issues with MPI+CUDA. Detected libs:\n$(join(ucx_libs, "\n"))"
end

κ_skew = 195
κ_symmetric = 149

start_year = 1962
simulation_length = 40

arch = GPU()

Nx = 720 # longitudinal direction 
Ny = 360 # meridional direction 
Nz = 100

z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
const z_surf = z_faces(Nz)

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

tracer_advection   = WENO(order=7)
momentum_advection = WENOVectorInvariant(order=5)
free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=40minutes)

@inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) =  2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
@inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters=25days)
catke_closure = ClimaOcean.Oceans.default_ocean_closure()
eddy_closure  = IsopycnalSkewSymmetricDiffusivity(; κ_skew, κ_symmetric, skew_flux_formulation=AdvectiveFormulation())
closure = (catke_closure, horizontal_viscosity, eddy_closure)

EN4_dir = joinpath(homedir(), "EN4_data")
mkpath(EN4_dir)

start_date = DateTime(start_year, 1, 1)
end_date = start_date + Year(simulation_length)
simulation_period = Dates.value(Second(end_date - start_date))

@info "Settting up salinity restoring..."
@inline mask(x, y, z, t) = z >= z_surf - 1
Smetadata = Metadata(:salinity; dataset=EN4Monthly(), dir=EN4_dir, start_date, end_date)
FS = DatasetRestoring(Smetadata, grid; rate = 1/30days, mask, time_indices_in_memory = 10)

ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing = (; S = FS),
                         closure)

@info "Built ocean model $(ocean)"

set!(ocean.model, T=Metadatum(:temperature; dataset=EN4Monthly(), date=start_date, dir=EN4_dir),
                  S=Metadatum(:salinity;    dataset=EN4Monthly(), date=start_date, dir=EN4_dir))

# Default sea-ice dynamics and salinity coupling are included in the defaults
# sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))
sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
@info "Built sea ice model $(sea_ice)"

ECCO_dir = joinpath(homedir(), "ECCO_data")
mkpath(ECCO_dir)
# Note that we are initializing sea ice thickness and concentration from ECCO data at the first date of ECCO dataset (1992-01-01)
set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir=ECCO_dir),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=ECCO_dir))
@info "Initialized sea ice fields with ECCO data"

jra55_dir = joinpath(homedir(), "JRA55_data")
mkpath(jra55_dir)
dataset = MultiYearJRA55()
backend = JRA55NetCDFBackend(100)

@info "Setting up prescribed atmosphere $(dataset)"
atmosphere = JRA55PrescribedAtmosphere(arch; dir=jra55_dir, dataset, backend, include_rivers_and_icebergs=true, start_date, end_date)
radiation  = Radiation()

@info "Built atmosphere model $(atmosphere)"

omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

@info "Built coupled model $(omip)"

omip = Simulation(omip, Δt=30minutes, stop_time=simulation_period) 
@info "Built simulation $(omip)"

FILE_DIR = joinpath(pwd(), "calibration_data", "half_degree_omip_spinup_skew_$(κ_skew)_symmetric_$(κ_symmetric)_$(start_year)_$(simulation_length)years")
mkpath(FILE_DIR)

b = buoyancy_field(ocean.model)
N² = Field(buoyancy_frequency(ocean.model))
h, ℵ = sea_ice.model.ice_thickness, sea_ice.model.ice_concentration

@inline is_northern_hemisphere(i, j, k, grid) = φnode(i, j, grid.underlying_grid, Center(), Center()) > 0
@inline is_southern_hemisphere(i, j, k, grid) = φnode(i, j, grid.underlying_grid, Center(), Center()) < 0

h_north = condition_operand(identity, h, is_northern_hemisphere, 0)
h_south = condition_operand(identity, h, is_southern_hemisphere, 0)
ℵ_north = condition_operand(identity, ℵ, is_northern_hemisphere, 0)
ℵ_south = condition_operand(identity, ℵ, is_southern_hemisphere, 0)
sea_ice_extent_north = Integral(h_north * ℵ_north, dims=(1, 2))
sea_ice_extent_south = Integral(h_south * ℵ_south, dims=(1, 2))

ocean_outputs = merge(prognostic_fields(ocean.model), (; b, N²))
sea_ice_outputs = merge(prognostic_fields(sea_ice.model), (; T = sea_ice.model.ice_thermodynamics.top_surface_temperature))
sea_ice_extent_outputs = (; north=sea_ice_extent_north, south=sea_ice_extent_south)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities, (; b, N²));
                                            schedule = TimeInterval(180days),
                                            filename = "$(FILE_DIR)/ocean_surface_fields",
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(ocean.model, sea_ice_outputs;
                                            schedule = TimeInterval(180days),
                                            filename = "$(FILE_DIR)/sea_ice_surface_fields",
                                            overwrite_existing = true)

save_times = start_date:Year(1):end_date
times = Dates.value.(Dates.Second.(save_times[2:end] .- start_date))
annual_times = SpecifiedTimes(times)

times_5year = start_date:Year(5):end_date
times_5year_vals = Dates.value.(Dates.Second.(times_5year[2:end] .- start_date))
five_year_times = SpecifiedTimes(times_5year_vals)

final_year_jan = end_date - Year(1)
final_year_feb = final_year_jan + Month(1)
final_year_mar = final_year_jan + Month(2)
final_year_apr = final_year_jan + Month(3)
final_year_may = final_year_jan + Month(4)
final_year_jun = final_year_jan + Month(5)
final_year_jul = final_year_jan + Month(6)
final_year_aug = final_year_jan + Month(7)
final_year_sep = final_year_jan + Month(8)
final_year_oct = final_year_jan + Month(9)
final_year_nov = final_year_jan + Month(10)
final_year_dec = final_year_jan + Month(11)

ocean.output_writers[:annual_snapshot] = JLD2Writer(ocean.model, ocean_outputs;
                                                    schedule = annual_times,
                                                    filename = "$(FILE_DIR)/ocean_annual_snapshot_fields",
                                                    overwrite_existing = true)

sea_ice.output_writers[:annual_snapshot] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                        schedule = annual_times,
                                                        filename = "$(FILE_DIR)/sea_ice_annual_snapshot_fields",
                                                        overwrite_existing = true)

sampling_window_1year = Dates.value(Dates.Second(end_date - (end_date - Year(1))))
sampling_window_5year = Dates.value(Dates.Second(end_date - (end_date - Year(5))))
sampling_window_10year = Dates.value(Dates.Second(end_date - (end_date - Year(10))))

ocean.output_writers[:average_1year] = JLD2Writer(ocean.model, ocean_outputs;
                                                  schedule = AveragedTimeInterval(simulation_period, window=sampling_window_1year),
                                                  filename = "$(FILE_DIR)/ocean_complete_fields_1year_average",
                                                  overwrite_existing = true)

ocean.output_writers[:average_10year] = JLD2Writer(ocean.model, ocean_outputs;
                                                   schedule = AveragedTimeInterval(simulation_period, window=sampling_window_10year),
                                                   filename = "$(FILE_DIR)/ocean_complete_fields_10year_average",
                                                   overwrite_existing = true)

sea_ice.output_writers[:average_1year] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                 schedule = AveragedTimeInterval(simulation_period, window=sampling_window_1year),
                                                 filename = "$(FILE_DIR)/sea_ice_complete_fields_1year_average",
                                                 overwrite_existing = true)

sea_ice.output_writers[:average_5year] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                 schedule = AveragedTimeInterval(simulation_period, window=sampling_window_5year),
                                                 filename = "$(FILE_DIR)/sea_ice_complete_fields_5year_average",
                                                 overwrite_existing = true)

sea_ice.output_writers[:average_10year] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                  schedule = AveragedTimeInterval(simulation_period, window=sampling_window_10year),
                                                  filename = "$(FILE_DIR)/sea_ice_complete_fields_10year_average",
                                                  overwrite_existing = true)

sea_ice.output_writers[:extent_jan] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_feb - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_feb - final_year_jan))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_jan",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_feb] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_mar - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_mar - final_year_feb))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_feb",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_mar] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_apr - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_apr - final_year_mar))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_mar",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_apr] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_may - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_may - final_year_apr))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_apr",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_may] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_jun - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_jun - final_year_may))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_may",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_jun] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_jul - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_jul - final_year_jun))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_jun",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_jul] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_aug - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_aug - final_year_jul))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_jul",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_aug] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_sep - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_sep - final_year_aug))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_aug",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_sep] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_oct - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_oct - final_year_sep))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_sep",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_oct] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_nov - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_nov - final_year_oct))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_oct",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_nov] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(final_year_dec - start_date)), 
                                                            window=Dates.value(Dates.Second(final_year_dec - final_year_nov))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_nov",
                                                 overwrite_existing = true)

sea_ice.output_writers[:extent_dec] = JLD2Writer(sea_ice.model, sea_ice_extent_outputs;
                                                 schedule = AveragedTimeInterval(Dates.value(Dates.Second(end_date - start_date)), 
                                                            window=Dates.value(Dates.Second(end_date - final_year_dec))),
                                                 filename = "$(FILE_DIR)/sea_ice_extent_dec",
                                                 overwrite_existing = true)

ocean.output_writers[:checkpointer] = Checkpointer(ocean.model;
                                                   schedule = five_year_times,
                                                   dir = FILE_DIR,
                                                   prefix = "ocean_checkpointer",
                                                   overwrite_existing = true)

sea_ice.output_writers[:checkpointer] = Checkpointer(sea_ice.model;
                                                      schedule = five_year_times,
                                                      dir = FILE_DIR,
                                                      prefix = "sea_ice_checkpointer",
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