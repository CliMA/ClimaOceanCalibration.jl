# One-degree OMIP simulation of ClimaOcean's `OneDegree()` configuration with the CATKE viscosity and the tracer
# and turbulent kinetic energy diffusivities multiplied by `diffusivity_scale`.

using ArgParse
using ClimaOcean
using ConstructionBase: setproperties
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, static_column_depthᶜᶜᵃ, inactive_cell
using Oceananigans.Models: buoyancy_operation
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
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

"""
    scale_diffusivities(catke, scale)

`catke` with its viscosity and its tracer and turbulent kinetic energy diffusivities multiplied by `scale`: every
mixing length is linear in the stability-function, convective and entrainment coefficients, and the maxima scale
with them. The dissipation length scale is unchanged.
"""
function scale_diffusivities(catke, scale)
    mixing_length = catke.mixing_length
    unscaled = (:Cˢ, :Cᵇ, :Cˢᵖ, :CRiᵟ, :CRi⁰)
    coefficients = filter(∉(unscaled), propertynames(mixing_length))
    scaled = (; (c => scale * getproperty(mixing_length, c) for c in coefficients)...)

    return CATKEVerticalDiffusivity(Oceananigans.TimeSteppers.time_discretization(catke);
                                    mixing_length = setproperties(mixing_length, scaled),
                                    catke.turbulent_kinetic_energy_equation,
                                    maximum_tracer_diffusivity = scale * catke.maximum_tracer_diffusivity,
                                    maximum_tke_diffusivity = scale * catke.maximum_tke_diffusivity,
                                    maximum_viscosity = scale * catke.maximum_viscosity,
                                    catke.minimum_tke,
                                    catke.minimum_convective_buoyancy_flux,
                                    catke.negative_tke_damping_time_scale,
                                    catke.tke_time_step)
end

const c = Center()
const f = Face()

"""
The depth below the free surface at which the buoyancy `b` has dropped by `Δb★` from its value at the depth `hʳ`,
interpolated between the two levels on either side of the threshold as de Boyer Montégut et al. (2004) do.
"""
@inline function mixed_layer_depthᶜᶜᵃ(i, j, _, grid, b, Δb★, hʳ)
    Nz = size(grid, 3)
    η = znode(i, j, Nz + 1, grid, c, c, f)
    zʳ = η - hʳ

    k⁺ = Nz
    for k in Nz:-1:1
        k⁺ = ifelse(znode(i, j, k, grid, c, c, c) ≥ zʳ, k, k⁺)
    end
    k⁻ = max(k⁺ - 1, 1)
    z⁺ = znode(i, j, k⁺, grid, c, c, c)
    z⁻ = znode(i, j, k⁻, grid, c, c, c)
    weight = clamp((zʳ - z⁻) / max(z⁺ - z⁻, eps(typeof(zʳ))), 0, 1)
    bʳ = @inbounds b[i, j, k⁻] + weight * (b[i, j, k⁺] - b[i, j, k⁻])

    h = static_column_depthᶜᶜᵃ(i, j, grid)
    z₀ = zʳ
    Δb₀ = zero(bʳ)
    mixed = true

    for k in k⁻:-1:1
        zₖ = znode(i, j, k, grid, c, c, c)
        Δb = bʳ - @inbounds b[i, j, k]
        wet = !inactive_cell(i, j, k, grid)
        crossing = mixed & wet & (Δb ≥ Δb★)
        h = ifelse(crossing, η - (z₀ + (zₖ - z₀) * (Δb★ - Δb₀) / (Δb - Δb₀)), h)
        mixed = mixed & wet & !crossing
        z₀ = ifelse(mixed, zₖ, z₀)
        Δb₀ = ifelse(mixed, Δb, Δb₀)
    end

    return h
end

configuration = OneDegree(; name, vertical_mixing = scale_diffusivities(OneDegree().ocean.vertical_mixing, diffusivity_scale))
@info configuration
@info configuration.ocean.vertical_mixing

simulation = omip_simulation(configuration; arch = GPU(), forcing_directory = arguments["forcing-directory"],
                             stop_time = arguments["years"] * 365days, diagnostics = false)

ocean = simulation.model.ocean.model
grid = ocean.grid
catke_fields = ocean.closure_fields[findfirst(closure -> closure isa CATKEVerticalDiffusivity, ocean.closure)]

mixed_layer_depth = KernelFunctionOperation{Center, Center, Nothing}(mixed_layer_depthᶜᶜᵃ, grid, buoyancy_operation(ocean),
                                                                     2.87e-4, 10.0)

writer_kw = (; including = Symbol[], dir = output_directory, overwrite_files = true,
               jld2_kw = Dict(:compress => ZstdFilter()))

simulation.output_writers[:diffusivities] = JLD2Writer(ocean, (; difvmo = catke_fields.κu, difvho = catke_fields.κc);
                                                       writer_kw..., schedule = AveragedTimeInterval(15days),
                                                       file_splitting = TimeInterval(360days),
                                                       filename = name * "_diffusivities")

simulation.output_writers[:mixed_layer] = JLD2Writer(ocean, (; mlotst_bracketing = mixed_layer_depth);
                                                     writer_kw..., schedule = AveragedTimeInterval(5days),
                                                     file_splitting = TimeInterval(360days),
                                                     filename = name * "_mixed_layer")

add_omip_diagnostics!(simulation; output_directory, prefix = name)

checkpoint_iteration(path) = parse(Int, match(r"iteration(\d+)\.jld2$", path)[1])
checkpoints = filter(contains("_checkpoint_iteration"), readdir(mkpath(output_directory); join = true))
pickup = isempty(checkpoints) ? false : argmax(checkpoint_iteration, checkpoints)

run!(simulation; pickup)
