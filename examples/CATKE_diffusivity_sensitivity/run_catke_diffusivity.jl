# One-degree OMIP simulation of ClimaOcean's `OneDegree()` configuration with the CATKE viscosity and the tracer
# and turbulent kinetic energy diffusivities multiplied by `diffusivity_scale`.

using ArgParse
using ClimaOcean
using Oceananigans
using Oceananigans.Units
using JLD2: ZstdFilter
using CUDA

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--diffusivity-scale"
        arg_type = Float64
        default = 1.0
    "--years"
        arg_type = Float64
        default = 20.0
    "--output-directory"
        arg_type = String
        default = "catke_diffusivity"
    "--forcing-directory"
        arg_type = String
        default = expanduser("~/JRA55_data")
end
arguments = parse_args(settings)

diffusivity_scale = arguments["diffusivity-scale"]
name = "catke_" * string(diffusivity_scale)
output_directory = joinpath(arguments["output-directory"], name)

include("scaled_catke.jl")

configuration = OneDegree(; name, vertical_mixing = scale_diffusivities(OneDegree().ocean.vertical_mixing, diffusivity_scale))
@info configuration
@info configuration.ocean.vertical_mixing

simulation = omip_simulation(configuration; arch = GPU(), forcing_directory = arguments["forcing-directory"],
                             stop_time = arguments["years"] * 365days, diagnostics = false)

ocean = simulation.model.ocean.model
grid = ocean.grid
catke_fields = ocean.closure_fields[findfirst(closure -> closure isa CATKEVerticalDiffusivity, ocean.closure)]

potential_buoyancy = KernelFunctionOperation{Center, Center, Center}(potential_buoyancyᶜᶜᶜ, grid, ocean.buoyancy.formulation,
                                                                     ocean.tracers.T, ocean.tracers.S)

mixed_layer_depth = KernelFunctionOperation{Center, Center, Nothing}(mixed_layer_depthᶜᶜᵃ, grid, potential_buoyancy, 2.87e-4, 10.0)

writer_kw = (; including = Symbol[], dir = output_directory, overwrite_files = true,
               jld2_kw = Dict(:compress => ZstdFilter()))

simulation.output_writers[:diffusivities] = JLD2Writer(ocean, (; difvmo = catke_fields.κu, difvho = catke_fields.κc);
                                                       writer_kw..., schedule = AveragedTimeInterval(15days),
                                                       file_splitting = TimeInterval(360days),
                                                       filename = name * "_diffusivities")

simulation.output_writers[:mixed_layer] = JLD2Writer(ocean, (; mlotst_potential = mixed_layer_depth);
                                                     writer_kw..., schedule = AveragedTimeInterval(5days),
                                                     file_splitting = TimeInterval(360days),
                                                     filename = name * "_mixed_layer")

add_omip_diagnostics!(simulation; output_directory, prefix = name)

checkpoint_iteration(path) = parse(Int, match(r"iteration(\d+)\.jld2$", path)[1])
checkpoints = filter(contains("_checkpoint_iteration"), readdir(mkpath(output_directory); join = true))
pickup = isempty(checkpoints) ? false : argmax(checkpoint_iteration, checkpoints)

run!(simulation; pickup)
