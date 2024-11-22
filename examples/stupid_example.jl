using Statistics
using OrderedCollections
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using GLMakie

using EnsembleKalmanProcesses.ParameterDistributions

prior1 = constrained_gaussian("C1", 1.0, 0.2, 0, 2)
prior2 = constrained_gaussian("C2", 0.5, 0.2, 0, 2)

prior = combine_distributions([prior1, prior2])

forward_map(C, index) = repeat(C, index)

# using EnsembleKalmanProcesses.ParameterDistributions
y1 = Observation(Dict("samples" => data, "covariances" => 1e-6 * EKP.I, "names" => "surface temperature"))
y2 = Observation(Dict("samples" => data[1:2:end], "covariances" => 1e-6 * EKP.I, "names" => "surface temperature"))
y3 = Observation(Dict("samples" => data, "covariances" => 1e-6 * EKP.I, "names" => "surface temperature"))
y4 = Observation(Dict("samples" => data, "covariances" => 1e-6 * EKP.I, "names" => "surface temperature"))
y5 = Observation(Dict("samples" => data, "covariances" => 1e-6 * EKP.I, "names" => "surface temperature"))

minibatcher = RandomFixedSizeMinibatcher(2)
y_series = ObservationSeries(Dict("observations" => [y1, y2, y3, y4, y5], "minibatcher" => minibatcher))
simulations = [sim1, sim2, sim3, sim4, sim5]

Ne = 10 # ensemble members
initial_ensemble = EKP.construct_initial_ensemble(prior, Ne)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y_series, TransformInversion())

B = get_current_minibatch(ensemble_kalman_process)
iteration_data = [OrderedDict() for b in B]

ℂ = get_ϕ_final(prior, ensemble_kalman_process)
B = get_current_minibatch(ensemble_kalman_process)

# Can also try: asyncmap(1:Ne, ntasks=10) do e to compute forward simulations in parallel  
for e = 1:Ne
    for (n, b) in enumerate(B)
        sim_e, data_e = forward_map(ℂ[:, e], simulations[b], b)
        iteration_data[b][e] = data_e
    end
end

#=
fig = Figure()
ax = Axis(fig[1, 1])

for data_e in values(iteration_data)
    lines!(ax, data_e .- data1)
end

display(fig) 
sleep(0.1)
=#

# G_ens .= hcat(values(iteration_data[1])...)
# EKP.update_ensemble!(ensemble_kalman_process, G_ens)
