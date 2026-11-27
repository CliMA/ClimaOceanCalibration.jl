# Mixed-layer depth recomputed from the saved 15-day-mean buoyancy with the de Boyer Montégut et al. (2004)
# criterion, interpolating between the two levels that bracket the threshold, and with the interpolation of the
# online diagnostic (between the 10 m reference and the first level past the threshold).

using ArgParse
using ClimaOcean
using ClimaOcean: OMIPOutput
using CairoMakie
using GeoMakie
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using Dates: month

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--runs-directory"
        arg_type = String
        default = expanduser("~/gm_tapering_sensitivity")
    "--figure-directory"
        arg_type = String
        default = expanduser("~/gm_tapering_sensitivity/figures")
    "--max-slopes"
        arg_type = Float64
        nargs = '+'
        default = [1e-3, 1e-2, 1e-1]
    "--reference-slope"
        arg_type = Float64
        default = 1e-2
    "--first-year"
        arg_type = Float64
        default = 15.0
    "--last-year"
        arg_type = Float64
        default = 20.0
end
arguments = parse_args(settings)

figure_directory = mkpath(arguments["figure-directory"])
max_slopes = arguments["max-slopes"]
reference_slope = arguments["reference-slope"]

const Ext = Base.get_extension(ClimaOcean, :ClimaOceanMakieExt)

run_name(max_slope) = "taper_" * string(max_slope)
run_label(max_slope) = @sprintf("Sₘₐₓ = %g (%g×)", max_slope, max_slope / reference_slope)

averaging_window = (arguments["first-year"] * 365days + 1, arguments["last-year"] * 365days + 1)
outputs = [OMIPOutput(joinpath(arguments["runs-directory"], run_name(s)); prefix = run_name(s), label = run_label(s), averaging_window)
           for s in max_slopes]
reference = outputs[findfirst(==(reference_slope), max_slopes)]

const Δb★ = 2.87e-4
const reference_depth = 10.0

"""
The depth below which the buoyancy `b` at cell centers `z` (bottom to top) has dropped by Δb★ from its value at
10 m, searching down to level `k_bottom`.
"""
function mixed_layer_depth(b, z, k_bottom, bracketing)
    zʳ = - reference_depth
    k⁺ = findfirst(≥(zʳ), z)
    k⁻ = k⁺ - 1
    k⁻ < k_bottom && return - z[k_bottom]
    bʳ = b[k⁻] + (zʳ - z[k⁻]) / (z[k⁺] - z[k⁻]) * (b[k⁺] - b[k⁻])

    for k in k⁻:-1:k_bottom
        Δb = bʳ - b[k]
        Δb < Δb★ && continue
        z₀, Δb₀ = (bracketing && k < k⁻) ? (z[k+1], bʳ - b[k+1]) : (zʳ, zero(Δb))
        return - (z₀ + (z[k] - z₀) * (Δb★ - Δb₀) / (Δb - Δb₀))
    end
    return - z[k_bottom]
end

"""Monthly climatologies of the mixed-layer depth from the 15-day buoyancy, with both interpolations."""
monthly_mixed_layer_depth(o) = Ext.cached(o, :monthly_mixed_layer_depth) do
    fts = Ext.timeseries(o, "fields", "bo")
    records = Ext.window_records(o, fts.times)
    months = month.(Ext.record_dates(o, fts.times[records]))
    mask = Ext.ocean_mask(o)
    z = Ext.depths(o)
    Nx, Ny, Nz = size(mask)
    k_bottom = [something(findfirst(view(mask, i, j, :)), Nz + 1) for i in 1:Nx, j in 1:Ny]

    sums = Dict(method => [zeros(Nx, Ny) for _ in 1:12] for method in (:bracketing, :diagnostic))
    counts = zeros(Int, 12)
    for (n, m) in zip(records, months)
        b = Ext.record_array(fts, n)
        Threads.@threads for j in 1:Ny
            for i in 1:Nx
                k_bottom[i, j] > Nz && continue
                column = view(b, i, j, :)
                sums[:bracketing][m][i, j] += mixed_layer_depth(column, z, k_bottom[i, j], true)
                sums[:diagnostic][m][i, j] += mixed_layer_depth(column, z, k_bottom[i, j], false)
            end
        end
        counts[m] += 1
    end

    surface = mask[:, :, end]
    Dict(method => [Ext.regrid_to_map(o, ifelse.(surface, sums[method][m] ./ counts[m], NaN)) for m in 1:12]
         for method in (:bracketing, :diagnostic))
end

latitudes, longitudes = Ext.map_latitudes, Ext.map_longitudes

regions = [("tropics, 20ᵒS–20ᵒN", (-20, 20), nothing),
           ("northern extratropics, 30ᵒN–60ᵒN", (30, 60), nothing),
           ("southern extratropics, 60ᵒS–30ᵒS", (-60, -30), nothing),
           ("Southern Ocean, 65ᵒS–50ᵒS", (-65, -50), nothing),
           ("subpolar North Atlantic, 50ᵒN–65ᵒN, 60ᵒW–10ᵒW", (50, 65), (300, 350))]

function regional_mean(map, (_, (φ₁, φ₂), λ_range))
    total = weight = 0.0
    for j in eachindex(latitudes), i in eachindex(longitudes)
        φ₁ ≤ latitudes[j] ≤ φ₂ || continue
        isnothing(λ_range) || λ_range[1] ≤ longitudes[i] ≤ λ_range[2] || continue
        isfinite(map[i, j]) || continue
        total += map[i, j] * cosd(latitudes[j])
        weight += cosd(latitudes[j])
    end
    return total / weight
end

observed = Ext.boyer_montegut_mixed_layer_depth()
extremum(maps, f) = f(cat(maps...; dims = 3); dims = 3)[:, :, 1]

columns = length(outputs) + 1
figure = Ext.omip_figure(columns, 3) do figure
    rows = (("winter maximum MLD, bracketing levels", :bracketing, maximum, (0, 600)),
            ("winter maximum MLD, interpolation from 10 m", :diagnostic, maximum, (0, 600)),
            ("summer minimum MLD, bracketing levels", :bracketing, minimum, (0, 80)))
    for (r, (title, method, f, colorrange)) in enumerate(rows)
        plot = Ext.global_map!(Ext.panel(figure, r, 1), extremum(observed, f);
                               title = "de Boyer Montégut et al. (2004)", colormap = Ext.mixed_layer_colormap, colorrange)
        for (c, o) in enumerate(outputs)
            plot = Ext.global_map!(Ext.panel(figure, r, c + 1), extremum(monthly_mixed_layer_depth(o)[method], f);
                                   title = string(o.label, ": ", title), colormap = Ext.mixed_layer_colormap, colorrange)
        end
        Ext.row_colorbar!(figure, r, columns, plot, "m")
    end
end
save(joinpath(figure_directory, "mld_recomputed_maps.png"), figure; px_per_unit = 1.5)

figure = Ext.omip_figure(length(regions), 1; column_width = 420, row_height = 380) do figure
    axes = map(enumerate(regions)) do (r, region)
        axis = Ext.climatology_axis(figure[1, r]; title = region[1], ylabel = "MLD (m)")
        lines!(axis, 1:12, [regional_mean(observed[m], region) for m in 1:12]; Ext.observation_line...,
               label = "de Boyer Montégut et al. (2004)")
        for (n, o) in enumerate(outputs)
            maps = monthly_mixed_layer_depth(o)
            Ext.run_lines!(axis, 1:12, [regional_mean(maps[:bracketing][m], region) for m in 1:12], n, o.label)
            lines!(axis, 1:12, [regional_mean(maps[:diagnostic][m], region) for m in 1:12];
                   color = (Ext.run_color(n), 0.6), linestyle = :dot, linewidth = 1.5)
        end
        axis
    end
    Legend(figure[2, 1:length(regions)], axes[1]; orientation = :horizontal, framevisible = false, merge = true)
end
save(joinpath(figure_directory, "mld_recomputed_seasonal_cycle.png"), figure; px_per_unit = 1.5)

open(joinpath(figure_directory, "mld_recomputed.csv"), "w") do io
    println(io, join(["run", "interpolation", ("$(replace(name, ", " => " ")) $(s) MLD (m)" for (name, _, _) in regions
                                               for s in ("annual-mean", "max-month", "min-month"))...], ","))
    for o in outputs, method in (:bracketing, :diagnostic)
        maps = monthly_mixed_layer_depth(o)[method]
        values = [f([regional_mean(maps[m], region) for m in 1:12]) for region in regions for f in (mean, maximum, minimum)]
        println(io, join([o.label, string(method), (@sprintf("%.4g", v) for v in values)...], ","))
    end
    values = [f([regional_mean(observed[m], region) for m in 1:12]) for region in regions for f in (mean, maximum, minimum)]
    println(io, join(["de Boyer Montégut et al. (2004)", "", (@sprintf("%.4g", v) for v in values)...], ","))
end
@info "Saved the recomputed mixed-layer depth"
