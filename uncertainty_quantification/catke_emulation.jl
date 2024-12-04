using JLD2, GLMakie, Statistics, ProgressBars, Random, Enzyme

## Abstractions

struct TwoLayerNetwork{M, V}
    W1::M
    b1::V
    W2::M
    b2::V
end

struct FourLayerNetwork{M, V}
    W1::M
    b1::V
    W2::M
    b2::V
    W3::M
    b3::V
    W4::M
    b4::V
end

struct NLayerNetwork{W, B}
    weights :: W
    biases :: B
end

function zero!(dnn::NLayerNetwork)
    for w in dnn.weights
        w .= 0
    end
    for b in dnn.biases
        b .= 0
    end
    return nothing
end

function zero!(dnn::TwoLayerNetwork)
    dnn.W1 .= 0.0
    dnn.b1 .= 0.0
    dnn.W2 .= 0.0
    dnn.b2 .= 0.0
    return nothing
end

function zero!(dnn::FourLayerNetwork)
    dnn.W1 .= 0.0
    dnn.b1 .= 0.0
    dnn.W2 .= 0.0
    dnn.b2 .= 0.0
    dnn.W3 .= 0.0
    dnn.b3 .= 0.0
    dnn.W4 .= 0.0
    dnn.b4 .= 0.0
    return nothing
end

function update!(nn::TwoLayerNetwork, dnn::TwoLayerNetwork, η)
    nn.W1 .-= η .* dnn.W1
    nn.b1 .-= η .* dnn.b1
    nn.W2 .-= η .* dnn.W2
    nn.b2 .-= η .* dnn.b2
    return nothing
end

function update!(nn::NLayerNetwork, dnn::NLayerNetwork, η)
    for (w, dw) in (nn.weights, dnn.weights)
        @. w -= η * dw
    end

    for (b, db) in (nn.biases, dnn.biases)
        @. b -= η * db
    end

    return nothing
end



function update!(nn::FourLayerNetwork, dnn::FourLayerNetwork, η)
    nn.W1 .-= η .* dnn.W1
    nn.b1 .-= η .* dnn.b1
    nn.W2 .-= η .* dnn.W2
    nn.b2 .-= η .* dnn.b2
    nn.W3 .-= η .* dnn.W3
    nn.b3 .-= η .* dnn.b3
    nn.W4 .-= η .* dnn.W4
    nn.b4 .-= η .* dnn.b4
    return nothing
end

swish(x) = x / (1 + exp(-x))
activate(x) = tanh(x) 

node(w, b, x) = activate.(w * x .+ b)
predict(nn::TwoLayerNetwork, x) = nn.W2 * node(nn.W1, nn.b1, x) .+ nn.b2

function predict(nn::FourLayerNetwork, x0)
    x1 = activate.(nn.W1 * x0 .+ nn.b1)
    x2 = activate.(nn.W2 * x1 .+ nn.b2)
    x3 = activate.(nn.W3 * x2 .+ nn.b3)
    return nn.W4 * x3 + nn.b4
end

## define functor 
(nn::TwoLayerNetwork)(x) = predict(nn, x)
(nn::FourLayerNetwork)(x) = predict(nn, x)
chunk_list(list, n, N) = [list[i:min(N, i+n-1)] for i in 1:n:length(list)]

function loss(nn::Union{TwoLayerNetwork, FourLayerNetwork}, x, y)
    ŷ = similar(y)
    for i in eachindex(ŷ)
        ŷ[i] = predict(nn, x[i])[1]
    end
    return mean((y .- ŷ) .^ 2)
end

# Load Data 
jlfile = jldopen("catke_parameters.jld2", "r")
θ = jlfile["parameters"]
y = jlfile["objectives"]
close(jlfile)
θa = reshape(θ, (size(θ)[1] * size(θ)[2], size(θ)[3]))
ya = reshape(y, (size(y)[1] * size(y)[2]))

ii = findall(y -> y < 100, ya)
training = ii[1:2:end]
testing  = ii[2:2:end]
yr = ya[training]
θr = θa[training, :]

yt = ya[testing]
θt = θa[testing, :]

M = size(θr)[1]
Mᴾ = size(θr)[2]

# Define Network
nodes = 256
inputs = 23

# input layer
Wi = randn(nodes, inputs)
bi = randn(nodes)

# inner layer
W1 = randn(nodes, nodes)
b1 = randn(nodes)
W2 = randn(nodes, nodes)
b2 = randn(nodes)

# output layer
Wo = randn(1, nodes)
bo = randn(1)

nn2 = TwoLayerNetwork(Wi, bi, Wo, bo)
nn4 = FourLayerNetwork(Wi, bi, W1, b1, W2, b2, Wo, bo)
dnn2 = deepcopy(nn2)
dnn4 = deepcopy(nn4)

# nn = nn2
# dnn = dnn2

nn = nn4
dnn = dnn4

# Optimize with Gradient Descent and Learning rate 1e-5
Random.seed!(1234) 
batchsize = 10
loss_list = Float64[]
epochs = 10
for i in ProgressBar(1:epochs)
    shuffled_list = chunk_list(shuffle(1:M), batchsize, M)
    @show shuffled_list
    loss_value = 0.0
    for permuted_list in ProgressBar(shuffled_list)
        θbatch = [θr[x, :] for x in permuted_list]
        ybatch = yr[permuted_list]
        zero!(dnn)
        autodiff(Enzyme.Reverse, loss, Active, DuplicatedNoNeed(nn, dnn), Const(θbatch), Const(ybatch))
        update!(nn, dnn, 1e-5)
        loss_value += loss(nn, θbatch, ybatch)
    end
    push!(loss_list, loss_value)
end

## evaluate
# index = 1000
# nn(θr[index, :])
# yr[index]

Ni = size(θt, 1)
Gt = [nn(θt[i, :])[1] for i = 1:size(θt, 1)]

using GLMakie

fig = Figure()
ax = Axis(fig[1, 1], ylabel="Emulation error, |ℕℕ(ℂ) - Φ(ℂ)| ./ Φ(ℂ)", xlabel="EKI objective Φ(ℂ)")
scatter!(ax, yt, abs.(Gt .- yt) ./ yt, color=(:blue, 0.1))
