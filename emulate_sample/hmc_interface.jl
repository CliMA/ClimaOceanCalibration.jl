struct LogDensity{N, S, M}
    logp::N
    regularization::S 
    scale::M
    scale_regularization::M
end

# Negative sign if the network represents the potential function
# Note: regularization should be negative semi-definite
function (logp::LogDensity{T})(θ) where T <: SimpleNetwork 
    return -logp.logp(θ)[1] * logp.scale + logp.regularization(θ) * logp.scale_regularization
end

function LogDensity(network::SimpleNetwork)
    regularization(x) = 0.0
    return LogDensity(network, regularization, 1.0, 1.0)
end

function LogDensity(network::SimpleNetwork, scale)
    regularization(x) = 0.0
    return LogDensity(network, regularization, scale, 1.0)
end

struct GradientLogDensity{N}
    logp::N 
    dθ::Vector{Float64}
end

function GradientLogDensity(logp::LogDensity{S}) where S <: SimpleNetwork
    dθ = zeros(size(logp.logp.W1, 2))
    return GradientLogDensity(logp, dθ)
end

function (∇logp::GradientLogDensity)(θ)
    ∇logp.dθ .= 0.0
    autodiff(Enzyme.Reverse, Const(∇logp.logp), Active, DuplicatedNoNeed(θ, ∇logp.dθ))
    return (∇logp.logp(θ),  copy(∇logp.dθ))
end

struct Regularizer{F}
    parameters::F
end
