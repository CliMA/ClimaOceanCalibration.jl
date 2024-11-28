using Enzyme

abstract type SimpleNetwork end

struct OneLayerNetwork{M, V} <: SimpleNetwork
    W1::M
    b1::V
    W2::M
    b2::V
end

function zero!(dnetwork::OneLayerNetwork)
    dnetwork.W1 .= 0.0
    dnetwork.b1 .= 0.0
    dnetwork.W2 .= 0.0
    dnetwork.b2 .= 0.0
    return nothing
end

function update!(network::SimpleNetwork, dnetwork::SimpleNetwork, η)
    network.W1 .-= η .* dnetwork.W1
    network.b1 .-= η .* dnetwork.b1
    network.W2 .-= η .* dnetwork.W2
    network.b2 .-= η .* dnetwork.b2
    return nothing
end

swish(x) = x / (1 + exp(-x))
activation_function(x) = tanh(x) 

function predict(network::OneLayerNetwork, x)
    return abs.(network.W2 * activation_function.(network.W1 * x .+ network.b1) .+ network.b2)
end

function predict(network::OneLayerNetwork, x, activation::Function)
    return abs.(network.W2 * activation.(network.W1 * x .+ network.b1) .+ network.b2)
end

function (network::SimpleNetwork)(x)
    return predict(network, x)
end