using ClimaOceanCalibration: diffusive_ocean_simulation, reschedule!
using Oceananigans
using Oceananigans.Units
using GLMakie

# Three degree by default, reaches ~2 SYPD on a laptop
simulation = diffusive_ocean_simulation(size=(120, 60, 10))

simulation.Δt = 1minute
simulation.stop_iteration = 10
run!(simulation)

ocean = simulation.model.ocean
grid = ocean.model.grid
T = ocean.model.tracers.T
heatmap(interior(T, :, :, size(grid, 3)))

parameters_names = (:Cᴷ, :Cᵃ, :Cᵇ)
calibrate_parameters!(simulation, parameter_names, priors)

