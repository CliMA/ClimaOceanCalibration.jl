using JLD2
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using StatsBase

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.Utilities


function main()

    data_file = joinpath(@__DIR__, "catke_parameters.jld2")

    # contains:
    loaded_data = load(data_file)
    objectives = loaded_data["objectives"]
    @info "objectives size $(size(objectives))"
    phys_parameters = loaded_data["parameters"]
    @info "parameters size $(size(phys_parameters))"
    #observations = loaded_data["observations"]
    #@info "observations size $size(observations)"
    # noise_covariance = loaded_data["noise_covariance"]
    # observations = loaded_data["observations"]
    n_iter, n_ens, n_param = size(phys_parameters)

    # (use priors to transform parameters into computational space)
    # get from initial ensemble
    #prior_mean = mean(parameters[1,:,:], dims=1)
    #prior_cov = cov(parameters[1,:,:], dims=1)
    #@info "det prior cov: $(det(prior_cov))"
    #@info "pos def prior cov?: $(isposdef(prior_cov))"
    prior_vec = [
        constrained_gaussian("CᵂwΔ", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("Cᵂu★", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("Cʰⁱc", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cʰⁱu", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cʰⁱe", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("CʰⁱD", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("Cˢ", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cˡᵒc", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cˡᵒu", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cˡᵒe", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("CˡᵒD", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("CRi⁰", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("CRiᵟ", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cᵘⁿc", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cᵘⁿu", 1.0, 0.5, 0.0, 2.0),
        constrained_gaussian("Cᵘⁿe", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("CᵘⁿD", 5.0, 2.5, 0.0, 10.0),
        constrained_gaussian("Cᶜc", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("Cᶜu", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("Cᶜe", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("CᶜD", 4.0, 2.0, 0.0, 8.0),
        constrained_gaussian("Cᵉc", 0.5, 0.25, 0.0, 1.0),
        constrained_gaussian("Cˢᵖ", 0.5, 0.25, 0.0, 1.0),
    ]
    prior = combine_distributions(prior_vec)

    # we map in the computational space
    parameters = zeros(size(phys_parameters))
    for i =1:size(phys_parameters,1)
        parameters[i,:,:] = transform_constrained_to_unconstrained(prior,phys_parameters[1,:,:]')'
    end
    
    
    ## CES on compressed data

    io_pair_iter = 10
    iter_train = 1:10:1+io_pair_iter*10
    @info "train on iterations: $(iter_train)"
    inputs = reduce(vcat, [parameters[i,:,:] for i in iter_train])
    outputs = reduce(vcat, objectives[iter_train,:])[:,:] 
    @info "training data input: $(size(inputs)) output: $(size(outputs))"
    input_output_pairs = PairedDataContainer(inputs, outputs, data_are_columns=false)

   # estimate the noise in the objective using trailing samples
    estimation_iter = 50
    Γ = zeros(1,1)
    Γ[1,1] = var(reduce(vcat, objectives[end - estimation_iter:end,:]))
    @info "estimated_noise: $Γ"
    
    truth = [minimum(objectives);]
    @info "data: $(truth)"
    
    @info "Finished inverse problem and data wrangling stage"
    #try GP JL
    gppackage = Emulators.GPJL()
    pred_type = Emulators.YType()

    mlt_methods = ["GP", "Scalar-RF"]
    
    mlt_method = mlt_methods[2]
    
    if mlt_method == "GP"
        mlt = GaussianProcess(
        gppackage;
        kernel = nothing, # use default squared exponential kernel
        prediction_type = pred_type,
        noise_learn = false,
    )
    elseif mlt_method == "Scalar-RF"
        overrides = Dict(
            "verbose" => true,
            "n_features_opt" => 20,
            "train_fraction" => 0.95,
            "n_iteration" => 20,
            "cov_sample_multiplier" => 1.0,
            "n_ensemble" => 200, #40*n_dimensions,
            "n_cross_val_sets" => 3,
        )

        rank = 5
        nugget=1e-6
        kernel_structure = SeparableKernel(LowRankFactor(rank, nugget), OneDimFactor())
        
        n_features = 500

        mlt = ScalarRandomFeatureInterface(
            n_features,
            n_param,
            kernel_structure = kernel_structure,
            optimizer_options = deepcopy(overrides),
        )

    end
    

    # Fit an emulator to the data
    emulator = Emulator(
        mlt,
        input_output_pairs;
        obs_noise_cov = Γ
    )

    # Optimize the GP hyperparameters for better fit
    optimize_hyperparameters!(emulator)

    @info "Finished Emulation stage"

    # test the emulator against some other trajectory data
    min_iter_test = maximum(iter_train) + 1
    iter_test = min_iter_test:min_iter_test + 10
    
    @info "test on iterations: $(iter_test)"
    test_inputs = reduce(vcat, [parameters[i,:,:] for i in iter_test])
    test_outputs = reduce(vcat, objectives[iter_test,:])[:,:] 
    @info "testing data input: $(size(test_inputs)) output: $(size(test_outputs))"
    pred_mean_test = zeros(length(iter_test)*n_ens,1)    
    for i in 1:size(test_inputs,1)
        pred_mean_test[i,:] = Emulators.predict(emulator, test_inputs[i:i,:]', transform_to_real = true)[1] 
    end
    test_error = norm(pred_mean_test - test_outputs)/size(test_inputs,1)
    @info "average L^2 test_error $(test_error)"

    pred_mean_train = zeros(io_pair_iter*n_ens,1)    
    for i in 1:size(get_inputs(input_output_data),2)
        pred_mean_test[i,:] = Emulators.predict(emulator, get_inputs(input_output_pairs)[:,i:i], transform_to_real = true)[1]
    end
    train_error = norm(pred_mean_train - get_outputs(input_output_pairs))/size(inputs,1)
    @info "average L^2 train_error $(train_error)"
    
    # determine a good step size
    u0 = vec(mean(get_inputs(input_output_pairs)[end,:], dims = 2))
    println("initial parameters: ", u0)
    yt_sample = truth
    mcmc = MCMCWrapper(RWMHSampling(), yt_sample, prior, emulator, init_params=u0)
    
    new_step = optimize_stepsize(mcmc; init_stepsize = 1.0, N = 5000, discard_initial = 0)
    chain = MarkovChainMonteCarlo.sample(mcmc, 300_000; stepsize = new_step, discard_initial = 2_000)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    
    println("Finished Sampling stage")
    # extract some statistics
    post_mean = mean(posterior)
    post_cov = cov(posterior)
    println("mean in of posterior samples (taken in comp space)")
    println(post_mean)
    println("covariance of posterior samples (taken in comp space)")
    println(post_cov)
    println("transformed posterior mean from comp to phys space")
    println(transform_unconstrained_to_constrained(posterior, post_mean))

    # map to physical space
    posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
    transformed_posterior_samples =
        mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)
    println("mean of posterior samples (taken in phys space)")
    println(mean(transformed_posterior_samples, dims = 2))
    println("cov of posterior samples (taken in phys space)")
    println(cov(transformed_posterior_samples, dims = 2))

    
end

main()
