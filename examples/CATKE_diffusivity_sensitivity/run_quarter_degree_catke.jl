# Quarter-degree OMIP simulation of ClimaOcean's `QuarterDegree()` configuration, which has no Gent-McWilliams
# closure, with the CATKE viscosity and the tracer and turbulent kinetic energy diffusivities multiplied by
# `diffusivity_scale`.

using ArgParse
using ClimaOcean
using ClimaOcean: meridional_volume_flux
using Oceananigans
using Oceananigans.Units
using JLD2: jldopen, ZstdFilter
using CUDA

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--diffusivity-scale"
        arg_type = Float64
        default = 1.0
    "--stop-day"
        arg_type = Float64
        default = 7300.0
    "--field-output"
        help = "also write the three-dimensional fields and the CATKE diffusivities"
        action = :store_true
    "--surface-interval-days"
        arg_type = Float64
        default = 5.0
    "--field-interval-days"
        arg_type = Float64
        default = 15.0
    "--checkpoint-interval"
        help = "iterations between checkpoints"
        arg_type = Int
        default = 4000
    "--output-directory"
        arg_type = String
        default = "quarter_degree_catke"
    "--forcing-directory"
        arg_type = String
        default = expanduser("~/JRA55_data")
end
arguments = parse_args(settings)

diffusivity_scale = arguments["diffusivity-scale"]
name = "quarter_catke_" * string(diffusivity_scale)
output_directory = mkpath(joinpath(arguments["output-directory"], name))

include("scaled_catke.jl")

configuration = QuarterDegree(; name, vertical_mixing = scale_diffusivities(QuarterDegree().ocean.vertical_mixing, diffusivity_scale))
@info configuration
@info configuration.ocean.vertical_mixing

simulation = omip_simulation(configuration; arch = GPU(), forcing_directory = arguments["forcing-directory"],
                             stop_time = arguments["stop-day"] * days, diagnostics = false)

coupled_model = simulation.model
ocean = coupled_model.ocean.model
sea_ice = coupled_model.sea_ice.model
grid = ocean.grid
Nz = size(grid, 3)
T, S, e = ocean.tracers.T, ocean.tracers.S, ocean.tracers.e
u, v, w = ocean.velocities
net_fluxes = coupled_model.interfaces.net_fluxes.ocean

potential_buoyancy = KernelFunctionOperation{Center, Center, Center}(potential_buoyancyᶜᶜᶜ, grid, ocean.buoyancy.formulation, T, S)
mixed_layer_depth = KernelFunctionOperation{Center, Center, Nothing}(mixed_layer_depthᶜᶜᵃ, grid, potential_buoyancy, 2.87e-4, 10.0)

surface_outputs = (; tos = view(T, :, :, Nz), sos = view(S, :, :, Nz), zos = ocean.free_surface.displacement,
                     tauuo = net_fluxes.u, tauvo = net_fluxes.v, hfds = net_fluxes.T, wfo = net_fluxes.S,
                     siconc = sea_ice.ice_concentration, sithick = sea_ice.ice_thickness)

checkpoint_prefix = name * "_checkpoint"
checkpoint_iteration(path) = parse(Int, match(r"iteration(\d+)\.jld2$", path)[1])
is_readable(path) = try jldopen(file -> haskey(file, "simulation"), path) catch; false end

"""The checkpoints of this run, oldest first."""
function checkpoints()
    paths = filter(contains(checkpoint_prefix * "_iteration"), readdir(output_directory; join = true))
    return sort(paths; by = checkpoint_iteration)
end

# A run that is stopped while it writes a checkpoint leaves a file that cannot be read
function latest_readable_checkpoint()
    readable = filter(is_readable, checkpoints())
    return isempty(readable) ? false : last(readable)
end

pickup = latest_readable_checkpoint()

# The writers serialize against the ocean model. A run that continues from a checkpoint appends to its files.
# Every writer needs its own splitting schedule, because the schedule records when it last split.
writer_kw() = (; including = Symbol[], dir = output_directory, overwrite_files = pickup == false,
                 file_splitting = TimeInterval(360days), jld2_kw = Dict(:compress => ZstdFilter()))

surface_interval = arguments["surface-interval-days"] * days
field_interval = arguments["field-interval-days"] * days

simulation.output_writers[:surface] = JLD2Writer(ocean, surface_outputs; writer_kw()..., schedule = AveragedTimeInterval(surface_interval),
                                                 filename = name * "_surface")

simulation.output_writers[:mixed_layer] = JLD2Writer(ocean, (; mlotst_potential = mixed_layer_depth); writer_kw()...,
                                                     schedule = AveragedTimeInterval(surface_interval),
                                                     filename = name * "_mixed_layer")

if arguments["field-output"]
    catke_fields = ocean.closure_fields[findfirst(closure -> closure isa CATKEVerticalDiffusivity, ocean.closure)]
    vvol = KernelFunctionOperation{Center, Face, Center}(meridional_volume_flux, grid, v)

    simulation.output_writers[:fields] = JLD2Writer(ocean, (; to = T, so = S, uo = u, tke = e, vvol); writer_kw()...,
                                                    schedule = AveragedTimeInterval(field_interval),
                                                    filename = name * "_fields")

    simulation.output_writers[:diffusivities] = JLD2Writer(ocean, (; difvmo = catke_fields.κu, difvho = catke_fields.κc); writer_kw()...,
                                                           schedule = AveragedTimeInterval(field_interval),
                                                           filename = name * "_diffusivities")
end

checkpoint_interval = arguments["checkpoint-interval"]
simulation.output_writers[:checkpointer] = Checkpointer(coupled_model; dir = output_directory, prefix = checkpoint_prefix,
                                                        schedule = IterationInterval(checkpoint_interval))

# Only the latest checkpoint stays on disk. The older ones go one iteration after it is written.
remove_old_checkpoints(simulation) = foreach(rm, checkpoints()[1:end-1])
add_callback!(simulation, remove_old_checkpoints, IterationInterval(checkpoint_interval; offset = 1); name = :remove_old_checkpoints)

# The first file split renames the unsplit file to part 1. A run that stopped after that split, and continues
# from a checkpoint written before it, needs the unsplit file back.
if pickup != false
    parts = jldopen(pickup) do file
        writers = file["simulation/output_writers"]
        Dict(writer => writers[writer]["part"] for writer in keys(writers) if haskey(writers[writer], "part"))
    end

    for (writer, part) in parts
        unsplit = simulation.output_writers[Symbol(writer)].filepath
        first_part = replace(unsplit, r"\.jld2$" => "_part1.jld2")
        part == 1 && isfile(first_part) && mv(first_part, unsplit)
    end
end

run!(simulation; pickup, checkpoint_at_end = true)
