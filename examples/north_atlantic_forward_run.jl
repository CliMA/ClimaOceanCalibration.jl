using ClimaOceanCalibration: diffusive_ocean_simulation, reschedule!, reset_coupled_simulation!
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Statistics
using OrderedCollections

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    CATKEMixingLength,
    CATKEEquation

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using GLMakie

# TODO:
# * omit land in the observation vector
# * add salinity to observation vector
# * make it so we can modify a simulation in place rather than building a new one
# * try a regional case?
# * use actual ECCO data in loss function not just perfect model

EKP = EnsembleKalmanProcesses

# Three degree by default, reaches ~2 SYPD on a laptop
FT = Float64

function forward_run!(simulation, parameters, ensemble_index, iteration=0)
    @show Cˢ = parameters[1]
    @show CˡᵒD = parameters[2]

    mixing_length = CATKEMixingLength(; Cˢ, Cᵇ=0.4)
    turbulent_kinetic_energy_equation = CATKEEquation(; CˡᵒD)
    catke = CATKEVerticalDiffusivity(FT; mixing_length, turbulent_kinetic_energy_equation)

    closure = catke
    simulation.model.ocean.model.closure = closure
    reset_coupled_simulation!(simulation)

    simulation.Δt = 10minutes
    simulation.stop_time = 2days

    ocean_model = simulation.model.ocean.model
    T = ocean_model.tracers.T
    S = ocean_model.tracers.S
    outputs = (; T, S)
    filename = "north_atlantic_calibration_i$(iteration)_e$(ensemble_index).jld2"

    averaging_writer = JLD2OutputWriter(ocean_model, outputs; filename,
                                        schedule = AveragedTimeInterval(1day),
                                        overwrite_existing = true)

    simulation.output_writers[:time_averages] = averaging_writer

    run!(simulation)

    return nothing
end

longitude = (260, 360)
latitude = (-15, 65)
res = 2 # degree
Nx = 80 ÷ res
Ny = 90 ÷ res
Nz = 30

mixing_length = CATKEMixingLength(Cˢ=1.1, Cᵇ=0.01)
turbulent_kinetic_energy_equation = CATKEEquation(CˡᵒD=0.6)
catke = CATKEVerticalDiffusivity(FT; mixing_length, turbulent_kinetic_energy_equation)
simulation = diffusive_ocean_simulation(CPU(), FT; longitude, latitude,
                                        size = (Nx, Ny, Nz),
                                        closure = catke,
                                        progress_interval=10)

forward_run!(simulation, [1.1, 0.6], 0, 0)

filename = "north_atlantic_calibration_i0_e0.jld2"

Tt = FieldTimeSeries(filename, "T")
St = FieldTimeSeries(filename, "S")

fig = Figure()
ax = Axis(fig[1, 1])
T = simulation.model.ocean.model.tracers.T
Nz = size(T, 3)
heatmap!(ax, view(T, :, :, Nz))

