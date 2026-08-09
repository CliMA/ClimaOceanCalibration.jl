function loss(network::SimpleNetwork, x, y)
    ŷ = similar(y)
    for i in eachindex(ŷ)
        ŷ[i] = predict(network, x[i])[1]
    end
    return mean((y .- ŷ) .^ 2)
end

function chunk_list(list, n)
    return [list[i:min(i+n-1, length(list))] for i in 1:n:length(list)]
end

struct Adam{S, T, I}
    struct_copies::S
    parameters::T 
    t::I
end

function parameters(network::SimpleNetwork)
    network_parameters = []
    for names in propertynames(network)
        push!(network_parameters, getproperty(network, names)[:])
    end
    param_lengths = [length(params) for params in network_parameters]
    parameter_list = zeros(sum(param_lengths))
    start = 1
    for i in 1:length(param_lengths)
        parameter_list[start:start+param_lengths[i]-1] .= network_parameters[i]
        start += param_lengths[i]
    end
    return parameter_list
end

function set_parameters!(network::SimpleNetwork, parameters_list)
    param_lengths = Int64[]
    for names in propertynames(network)
        push!(param_lengths, length(getproperty(network, names)[:]))
    end
    start = 1
    for (i, names) in enumerate(propertynames(network))
        getproperty(network, names)[:] .= parameters_list[start:start+param_lengths[i]-1]
        start = start + param_lengths[i]
    end
    return nothing
end

function Adam(network::SimpleNetwork; α=0.001, β₁=0.9, β₂=0.999, ϵ=1e-8)
    parameters_list = (; α, β₁, β₂, ϵ)
    network_parameters = parameters(network)
    t = [1.0]
    θ  = deepcopy(network_parameters) .* 0.0
    gₜ = deepcopy(network_parameters) .* 0.0
    m₀ = deepcopy(network_parameters) .* 0.0
    mₜ = deepcopy(network_parameters) .* 0.0
    v₀ = deepcopy(network_parameters) .* 0.0
    vₜ = deepcopy(network_parameters) .* 0.0
    v̂ₜ = deepcopy(network_parameters) .* 0.0
    m̂ₜ = deepcopy(network_parameters) .* 0.0
    struct_copies = (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ)
    return Adam(struct_copies, parameters_list,  t)
end


function update!(adam::Adam, network::SimpleNetwork, dnetwork::SimpleNetwork)
    # unpack
    (; α, β₁, β₂, ϵ) = adam.parameters
    t = adam.t[1]
    (; θ, gₜ, m₀, mₜ, v₀, vₜ, v̂ₜ, m̂ₜ) = adam.struct_copies
    t = adam.t[1]
    # get gradient
    θ .= parameters(network)
    gₜ .= parameters(dnetwork)
    # update
    @. mₜ = β₁ * m₀ + (1 - β₁) * gₜ
    @. vₜ = β₂ * v₀ + (1 - β₂) * (gₜ .^2)
    @. m̂ₜ = mₜ / (1 - β₁^t)
    @. v̂ₜ = vₜ / (1 - β₂^t)
    @. θ = θ - α * m̂ₜ / (sqrt(v̂ₜ) + ϵ)
    # update parameters
    m₀ .= mₜ
    v₀ .= vₜ
    adam.t[1] += 1
    set_parameters!(network, θ)
    return nothing
end