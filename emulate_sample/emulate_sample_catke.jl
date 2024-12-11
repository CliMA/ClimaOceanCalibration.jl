using JLD2, Enzyme, ProgressBars, Random, Statistics, AdvancedHMC, GLMakie

Random.seed!(1234)
tic = time()
include("simple_networks.jl")
include("hmc_interface.jl")
include("optimization_utils.jl")

data_directory = ""
data_file = "catke_parameters.jld2"

jlfile = jldopen(data_directory * data_file, "r")
θ = jlfile["parameters"]
y = jlfile["objectives"]
close(jlfile)

θ̄ = mean(θ)
θ̃ = std(θ)
ymax = maximum(y)
ymin = minimum(y)
yshift = ymin # ymin / 2 #  
Δy =  ymax - ymin # 2 * std(y)  #
θr = (reshape(θ, (size(θ)[1] * size(θ)[2], size(θ)[3])) .- θ̄ ) ./ (2θ̃)
yr = (reshape(y, (size(y)[1] * size(y)[2])) .- yshift ) ./ Δy
M = size(θr)[1]
Mᴾ = size(θr)[2]

# Define Network
Nθ = size(θr, 2)
Nθᴴ = Nθ ÷ 2
W1 = randn(Nθᴴ, Nθ)
b1 = randn(Nθᴴ)
W2 = randn(1, Nθᴴ)
b2 = randn(1)
W3 = randn(1, Nθ)
b3 = randn(1)

network = OneLayerNetworkWithLinearByPass(W1, b1, W2, b2, W3, b3)
dnetwork = deepcopy(network)
smoothed_network = deepcopy(network)

## Emulate
adam = Adam(network)
batchsize = 100
loss_list = Float64[]
test_loss_list = Float64[]
epochs = 100
network_parameters = copy(parameters(network))
for i in ProgressBar(1:epochs)
    shuffled_list = chunk_list(shuffle(1:2:M), batchsize)
    shuffled_test_list = chunk_list(shuffle(2:2:M), batchsize)
    loss_value = 0.0
    N = length(shuffled_list)
    # Batched Gradient Descent and Loss Evaluation
    for permuted_list in ProgressBar(shuffled_list)
        θbatch = [θr[x, :] for x in permuted_list]
        ybatch = yr[permuted_list]
        zero!(dnetwork)
        autodiff(Enzyme.Reverse, loss, Active, DuplicatedNoNeed(network, dnetwork), Const(θbatch), Const(ybatch))
        update!(adam, network, dnetwork)
        loss_value += loss(network, θbatch, ybatch) / N
    end
    push!(loss_list, loss_value)
    # Test Loss
    loss_value = 0.0 
    N = length(shuffled_test_list)
    for permuted_list in shuffled_test_list
        θbatch = [θr[x, :] for x in permuted_list]
        ybatch = yr[permuted_list]
        loss_value += loss(network, θbatch, ybatch) / N
    end
    push!(test_loss_list, loss_value)
    # Weighted Averaging of Network
    m = 0.9
    network_parameters .= m * network_parameters + (1-m) * parameters(network)
    set_parameters!(smoothed_network, network_parameters)
end

loss_fig = Figure()
ax = Axis(loss_fig[1, 1]; title = "Log10 Loss", xlabel = "Epoch", ylabel = "Loss")
scatter!(ax, log10.(loss_list); color = :blue, label = "Training Loss")
scatter!(ax, log10.(test_loss_list); color = :red, label = "Test Loss")
axislegend(ax, position = :rt)
display(loss_fig)

## Sample
# Define logp and ∇logp and regularizer

initial_θ = copy(θr[argmin(yr), :])
mintheta = minimum(θr, dims = 1)[:]
maxtheta = maximum(θr, dims = 1)[:]
reg = Regularizer([mintheta, maxtheta, initial_θ])

function (regularizer::Regularizer)(x)
    if any(x .≤ regularizer.parameters[1])
        return -Inf
    elseif any(x .> regularizer.parameters[2])
        return -Inf
    else
        return -sum(abs.(x - regularizer.parameters[3]) ./ (regularizer.parameters[2] - regularizer.parameters[1]))
    end
    return 0.0
end

scale = 10 * Δy # 1/minimum(yr)
regularization_scale = 0.001/2 * scale

U = LogDensity(network, reg, scale, regularization_scale)
∇U = GradientLogDensity(U)

# HMC 
D = size(θr, 2)
n_samples = 10000
n_adapts = 1000

metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, GaussianKinetic(), U, ∇U)

initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true)

toc = time()
println("Elapsed time: $((toc - tic)/60) minutes")

# Plot
namelist = ["CᵂwΔ", "Cᵂu★", "Cʰⁱc", "Cʰⁱu", "Cʰⁱe", "CʰⁱD", "Cˢ", "Cˡᵒc", "Cˡᵒu", "Cˡᵒe", "CˡᵒD", "CRi⁰", "CRiᵟ", "Cᵘⁿc", "Cᵘⁿu", "Cᵘⁿe", "CᵘⁿD", "Cᶜc", "Cᶜu", "Cᶜe", "CᶜD", "Cᵉc", "Cˢᵖ"]
fig = Figure() 
Mp = 5
for i in 1:23
    ii = (i-1)÷Mp + 1
    jj = (i-1)%Mp + 1
    ax = Axis(fig[ii, jj]; title = namelist[i])
    v1 = ([sample[i] for sample in samples] .* 2θ̃) .+ θ̄
    hist!(ax, v1; bins = 50, strokewidth = 0, color = :blue, normalization = :pdf)
    xlims!(ax, -0.1, (reg.parameters[2][i]* 2θ̃ + θ̄) * 1.1)
    density!(ax, v1; color = (:red, 0.1), strokewidth = 3, strokecolor = :red)
end
display(fig)

imin = argmax([stat.log_density for stat in stats])
imax = argmin([stat.log_density for stat in stats])
network(samples[imin])
θ₀ = (initial_θ .* 2θ̃) .+ θ̄
((mean(samples) .* 2θ̃) .+ θ̄) - θ₀
((samples[imin] .* 2θ̃) .+ θ̄) - θ₀