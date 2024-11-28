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

θr = reshape(θ, (size(θ)[1] * size(θ)[2], size(θ)[3]))
yr = reshape(y, (size(y)[1] * size(y)[2]))
M = size(θr)[1]
Mᴾ = size(θr)[2]

# Define Network
Nθ = size(θr, 2)
Nθᴴ = Nθ * 10
W1 = randn(Nθᴴ, Nθ)
b1 = randn(Nθᴴ)
W2 = randn(1, Nθᴴ)
b2 = randn(1)

network = OneLayerNetwork(W1, b1, W2, b2)
dnetwork = deepcopy(network)

## Emulate
# Optimize with Gradient Descent and Learning rate 1e-5
batchsize = 10
loss_list = Float64[]
epochs = 10
for i in ProgressBar(1:epochs)
    shuffled_list = chunk_list(shuffle(1:M), batchsize)
    loss_value = 0.0
    N = length(shuffled_list)
    for permuted_list in ProgressBar(shuffled_list)
        θbatch = [θr[x, :] for x in permuted_list]
        ybatch = yr[permuted_list]
        zero!(dnetwork)
        autodiff(Enzyme.Reverse, loss, Active, DuplicatedNoNeed(network, dnetwork), Const(θbatch), Const(ybatch))
        update!(network, dnetwork, 1e-5)
        loss_value += loss(network, θbatch, ybatch) / N
    end
    push!(loss_list, loss_value)
end

## Sample
# HMC 
scale = 1.0e2

function regularize(x)
    if any(x .≤ 0.0)
        return -Inf
    elseif any(x .> 8.0)
        return -Inf
    else
        return 0.0
    end
    return 
end
regularization_scale = 1.0

U = LogDensity(network, regularize, scale, regularization_scale)
∇U = GradientLogDensity(U)

D = size(θr, 2)
initial_θ = copy(θr[end, :])
n_samples = 1000
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
fig = Figure() 
Mp = 5
for i in 1:23
    ii = (i-1)÷Mp + 1
    jj = (i-1)%Mp + 1
    ax = Axis(fig[ii, jj]; title = "Parameter $i")
    v1 = [sample[i] for sample in samples]
    hist!(ax, v1; bins = 50, strokewidth = 0, color = :blue, normalization = :pdf)
    density!(ax, v1; color = (:red, 0.1), strokewidth = 3, strokecolor = :red)
end
display(fig)