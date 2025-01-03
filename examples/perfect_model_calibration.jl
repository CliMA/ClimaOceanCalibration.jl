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

function forward_map(parameters, simulation=nothing)
    @show Cˢ = parameters[1]
    @show CˡᵒD = parameters[2]

    mixing_length = CATKEMixingLength(; Cˢ, Cᵇ=0.01)
    turbulent_kinetic_energy_equation = CATKEEquation(; CˡᵒD)
    catke = CATKEVerticalDiffusivity(FT; mixing_length, turbulent_kinetic_energy_equation)

    # gm = IsopycnalSkewSymmetricDiffusivity(FT; κ_skew=1000, κ_symmetric=1000)
    # closure = (gm, catke)
    
    closure = catke

    if isnothing(simulation)
        simulation = diffusive_ocean_simulation(CPU(), FT;
                                                size = (90, 25, 20),
                                                latitude = (-80, -20),
                                                closure,
                                                progress_interval=10)
    else
        simulation.model.ocean.model.closure = closure
        reset_coupled_simulation!(simulation)
    end

    @show simulation.model.ocean.model.closure

    simulation.Δt = 5minutes
    simulation.stop_time = 1day
    run!(simulation)

    grid = simulation.model.ocean.model.grid
    T = simulation.model.ocean.model.tracers.T
    mask_immersed_field!(T, NaN)
    surface_temperature = view(T, :, :, size(grid, 3))
    data = filter(Tᵢ -> !isnan(Tᵢ), surface_temperature[:])

    # Noise the data cause we hack
    data .+= 1e-12 .* randn(length(data))

    return simulation, data 
end

res = 4 # degree
Nx = 360 ÷ res
Ny = 60 ÷ res
Nz = 30

mixing_length = CATKEMixingLength(Cˢ=1.1, Cᵇ=0.01)
turbulent_kinetic_energy_equation = CATKEEquation(CˡᵒD=0.6)
catke = CATKEVerticalDiffusivity(FT; mixing_length, turbulent_kinetic_energy_equation)
simulation = diffusive_ocean_simulation(CPU(), FT;
                                        size = (Nx, Ny, Nz),
                                        latitude = (-80, -20),
                                        closure = catke,
                                        progress_interval=10)

sim1, data1 = forward_map([1.1, 0.6], simulation)

# ocean = simulation.model.ocean
# grid = ocean.model.grid
# heatmap(interior(T, :, :, size(grid, 3)))

using EnsembleKalmanProcesses.ParameterDistributions
prior1 = constrained_gaussian("C1", 1.0, 0.2, 0, 2)
prior2 = constrained_gaussian("C2", 0.5, 0.2, 0, 2)
prior = combine_distributions([prior1, prior2])

# using EnsembleKalmanProcesses.ParameterDistributions
metadata = Dict("samples" => data,
                "covariances" => 1e-6 * EKP.I,
                "names" => "surface temperature" )

y = Observation(metadata)
Ne = 10 # ensemble members
initial_ensemble = EKP.construct_initial_ensemble(prior, Ne)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, TransformInversion())

G_ens = zeros(length(data1), Ne)
iteration_data = OrderedDict()
Ni = 2 # number of iterations

for n in 1:Ni
    global G_ens, fig
    ℂ = get_ϕ_final(prior, ensemble_kalman_process)

    # Can also try: asyncmap(1:Ne, ntasks=10) do e to compute forward simulations in parallel  
    for e = 1:Ne
        sim_e, data_e = forward_map(ℂ[:, e], simulation)
        iteration_data[e] = data_e
    end

    fig = Figure()
    ax = Axis(fig[1, 1])

    for data_e in values(iteration_data)
        lines!(ax, data_e .- data1)
    end

    display(fig) 
    sleep(0.1)

    G_ens .= hcat(values(iteration_data)...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

