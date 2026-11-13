# Figures comparing the one-degree OMIP runs of run_gm_tapering.jl: buoyancy and isopycnal slopes, the
# Gent-McWilliams tapering and overturning, and the mixed-layer depth through the seasonal cycle.

using ArgParse
using ClimaOcean
using ClimaOcean: OMIPOutput
using CairoMakie
using GeoMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ
using Statistics
using Printf

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

runs_directory = arguments["runs-directory"]
figure_directory = mkpath(arguments["figure-directory"])
max_slopes = arguments["max-slopes"]
reference_slope = arguments["reference-slope"]
first_year, last_year = arguments["first-year"], arguments["last-year"]

const Ext = Base.get_extension(ClimaOcean, :ClimaOceanMakieExt)

run_name(max_slope) = "taper_" * string(max_slope)
run_label(max_slope) = @sprintf("Sₘₐₓ = %g (%g×)", max_slope, max_slope / reference_slope)

averaging_window = (first_year * 365days + 1, last_year * 365days + 1)
outputs = [OMIPOutput(joinpath(runs_directory, run_name(s)); prefix = run_name(s), label = run_label(s), averaging_window)
           for s in max_slopes]
reference = outputs[findfirst(==(reference_slope), max_slopes)]
perturbed = filter(o -> o !== reference, outputs)
max_slope_of = Dict(o.label => s for (o, s) in zip(outputs, max_slopes))
window_label = @sprintf("years %g–%g", first_year, last_year)

function save_figure(name, figure)
    save(joinpath(figure_directory, name * ".png"), figure; px_per_unit = 1.5)
    @info "Saved $name.png"
end

latitudes = Ext.map_latitudes
longitudes = Ext.map_longitudes

#####
##### Buoyancy and isopycnal slopes
#####

# The Mediterranean and Black Sea, whose dense outflow water would dominate the zonal means at 30ᵒN–47ᵒN
enclosed_sea_mask(o) = Ext.cached(o, :enclosed_sea_mask) do
    λ, φ = Ext.center_longitudes(o), Ext.center_latitudes(o)
    @. (30 ≤ φ ≤ 47) & ((λ ≥ 354.5) | (λ ≤ 42))
end

open_ocean_mask(o) = Ext.cached(() -> Ext.ocean_mask(o) .& .!enclosed_sea_mask(o), o, :open_ocean_mask)

open_ocean_zonal_mean(o, data) = Ext.zonal_mean(o, data, open_ocean_mask(o))

zonal_buoyancy(o) = Ext.cached(() -> open_ocean_zonal_mean(o, Ext.time_mean(o, "fields", "bo")), o, :zonal_buoyancy)

"""The slope −∂ᵧb̄ / ∂_z b̄ of the zonal-mean isopycnals."""
zonal_isopycnal_slope(o) = Ext.cached(o, :zonal_isopycnal_slope) do
    b = zonal_buoyancy(o)
    z = Ext.depths(o)
    Δy = 6.371e6 * deg2rad(latitudes[2] - latitudes[1])
    S = fill(NaN, size(b))
    for k in 2:length(z)-1, j in 2:length(latitudes)-1
        ∂y_b = (b[j+1, k] - b[j-1, k]) / 2Δy
        ∂z_b = (b[j, k+1] - b[j, k-1]) / (z[k+1] - z[k-1])
        S[j, k] = - ∂y_b / ∂z_b
    end
    S
end

wet_mean(a, x, b, y) = a & b ? (x + y) / 2 : a ? x : b ? y : zero(x)

"""The magnitude of the isopycnal slope of the time-mean buoyancy in every cell, Inf where it is unstratified."""
local_slope(o) = Ext.cached(o, :local_slope) do
    b = Ext.time_mean(o, "fields", "bo")
    mask = Ext.ocean_mask(o)
    grid = Ext.model_grid(o)
    Nx, Ny, Nz = size(b)
    S = fill(NaN, Nx, Ny, Nz)
    Threads.@threads for k in 1:Nz
        for j in 2:Ny-1, i in 1:Nx
            mask[i, j, k] || continue
            e, w = mod1(i + 1, Nx), mod1(i - 1, Nx)
            ∂x_b = wet_mean(mask[e, j, k], (b[e, j, k] - b[i, j, k]) / Δxᶠᶜᶜ(i + 1, j, k, grid),
                            mask[w, j, k], (b[i, j, k] - b[w, j, k]) / Δxᶠᶜᶜ(i, j, k, grid))
            ∂y_b = wet_mean(mask[i, j+1, k], (b[i, j+1, k] - b[i, j, k]) / Δyᶜᶠᶜ(i, j + 1, k, grid),
                            mask[i, j-1, k], (b[i, j, k] - b[i, j-1, k]) / Δyᶜᶠᶜ(i, j, k, grid))
            above = k < Nz && mask[i, j, k+1]
            below = k > 1 && mask[i, j, k-1]
            ∂z_b = wet_mean(above, above ? (b[i, j, k+1] - b[i, j, k]) / Δzᶜᶜᶠ(i, j, k + 1, grid) : 0.0,
                            below, below ? (b[i, j, k] - b[i, j, k-1]) / Δzᶜᶜᶠ(i, j, k, grid) : 0.0)
            S[i, j, k] = ∂z_b > 0 ? hypot(∂x_b, ∂y_b) / ∂z_b : Inf
        end
    end
    S
end

tapering_factor(S, max_slope) = isnan(S) ? S : min(1.0, max_slope^2 / S^2)

zonal_tapering(o) = Ext.cached(o, :zonal_tapering) do
    open_ocean_zonal_mean(o, tapering_factor.(local_slope(o), max_slope_of[o.label]))
end

zonal_log_slope(o) = Ext.cached(() -> open_ocean_zonal_mean(o, log10.(min.(local_slope(o), 1.0))), o, :zonal_log_slope)

section(o, data; overlay = nothing) = (latitudes, Ext.depths(o), data, overlay)

"""Like the extension's `sections_figure`, with the dashed overlay contoured at `overlay_levels`."""
function sections_figure(outputs, rows)
    columns = length(outputs)
    return Ext.omip_figure(columns, length(rows); column_width = 560, row_height = 330) do figure
        for (r, row) in enumerate(rows)
            plot = nothing
            for (c, o) in enumerate(outputs)
                x, z, data, overlay = row.data(o)
                plot = Ext.section_panel!(Ext.panel(figure, r, c), x, z, data; title = string(o.label, ": ", row.title),
                                          row.levels, row.colormap, overlay, row.overlay_levels,
                                          xlabel = "Latitude", row.depth_limit)
            end
            Ext.row_colorbar!(figure, r, columns, plot, row.label)
        end
    end
end

section_row(title, data, levels, colormap, label; overlay_levels = levels, depth_limit = 3000) =
    (; title, data, levels, colormap, label, overlay_levels, depth_limit)

symmetric_levels(values; quantile_level = 0.99) =
    let extent = quantile(abs.(filter(isfinite, values)), quantile_level)
        range(-extent, extent, length = 17)
    end

buoyancy_levels = range(-0.04, 0.02, length = 25)
isopycnal_levels = range(-0.04, 0.02, length = 13)
buoyancy_differences = reduce(vcat, vec(zonal_buoyancy(o) .- zonal_buoyancy(reference)) for o in perturbed)

save_figure("zonal_buoyancy",
    sections_figure(outputs, [
        section_row("zonal-mean b, control dashed", o -> section(o, zonal_buoyancy(o); overlay = zonal_buoyancy(reference)),
                    buoyancy_levels, :balance, "m s⁻²"; overlay_levels = isopycnal_levels),
        section_row("zonal-mean b − control", o -> section(o, zonal_buoyancy(o) .- zonal_buoyancy(reference)),
                    symmetric_levels(buoyancy_differences), :balance, "m s⁻²")]))

save_figure("isopycnal_slopes",
    sections_figure(outputs, [
        section_row("slope of zonal-mean isopycnals (×10⁻³)",
                    o -> section(o, 1e3 .* zonal_isopycnal_slope(o); overlay = zonal_buoyancy(o)),
                    range(-3, 3, length = 13), :balance, "−∂ᵧb̄ / ∂_z b̄ (×10⁻³)"; overlay_levels = isopycnal_levels),
        section_row("zonal mean of log₁₀ |local slope|",
                    o -> section(o, zonal_log_slope(o); overlay = zonal_buoyancy(o)),
                    range(-5, -1, length = 17), :viridis, "log₁₀ |S|"; overlay_levels = isopycnal_levels),
        section_row("zonal-mean tapering factor min(1, Sₘₐₓ² / S²)",
                    o -> section(o, zonal_tapering(o); overlay = zonal_buoyancy(o)),
                    range(0, 1, length = 11), :viridis, "tapering factor"; overlay_levels = isopycnal_levels)]))

slope_depths = (200, 500, 1000, 2000)

save_figure("isopycnal_slope_profiles",
    Ext.omip_figure(length(slope_depths), 1; column_width = 480, row_height = 380) do figure
        axes = map(enumerate(slope_depths)) do (c, depth)
            z = Ext.depths(reference)
            k = argmin(abs.(z .+ depth))
            axis = Axis(figure[1, c]; title = @sprintf("zonal-mean isopycnal slope at %d m", round(Int, -z[k])),
                        xlabel = "Latitude", ylabel = "−∂ᵧb̄ / ∂_z b̄ (×10⁻³)")
            for (n, o) in enumerate(outputs)
                Ext.run_lines!(axis, latitudes, 1e3 .* zonal_isopycnal_slope(o)[:, k], n, o.label)
            end
            ylims!(axis, -3, 3)
            axis
        end
        Legend(figure[2, 1:length(slope_depths)], axes[1]; orientation = :horizontal, framevisible = false)
    end)

#####
##### Overturning
#####

finite_or_zero(x) = ifelse(isfinite(x), x, zero(x))

global_streamfunction(o, names...) = Ext.cached(o, Symbol(:global_streamfunction, names...)) do
    Ny = size(Ext.surface_mask(o), 2)
    transport = sum(finite_or_zero.(Ext.time_mean(o, "fields", name)) for name in names)
    ψ = - cumsum(dropdims(sum(transport[:, 1:Ny, :]; dims = 1); dims = 1); dims = 2) ./ 1e6
    rows = 1:argmax(Ext.overturning_latitudes(o, Ny))
    (Ext.overturning_latitudes(o, Ny)[rows], Ext.depths(o), ψ[rows, :], nothing)
end

overturning_levels = range(-30, 30, length = 25)

save_figure("overturning",
    sections_figure(outputs, [
        section_row("GM bolus overturning ψ*", o -> global_streamfunction(o, "vvolgm"), overturning_levels, :balance, "Sv";
                    depth_limit = 5500),
        section_row("GM bolus overturning ψ*, upper 300 m", o -> global_streamfunction(o, "vvolgm"),
                    overturning_levels, :balance, "Sv"; depth_limit = 300),
        section_row("Eulerian overturning", o -> global_streamfunction(o, "vvol"), overturning_levels, :balance, "Sv";
                    depth_limit = 5500),
        section_row("residual overturning", o -> global_streamfunction(o, "vvol", "vvolgm"),
                    overturning_levels, :balance, "Sv"; depth_limit = 5500)]))

#####
##### Mixed-layer depth
#####

save_figure("mld_extrema_maps", ClimaOcean.mixed_layer_depth(outputs...))
save_figure("mld_zonal_extrema", ClimaOcean.zonal_mean_mixed_layer_depth(outputs...))

mld_extrema(o) = Ext.mixed_layer_depth_extrema_maps(o)

save_figure("mld_extrema_differences",
    Ext.maps_figure(perturbed, [
        Ext.map_row("winter maximum MLD − control", o -> mld_extrema(o).maximum .- mld_extrema(reference).maximum,
                    :balance, (-300, 300), "m"),
        Ext.map_row("summer minimum MLD − control", o -> mld_extrema(o).minimum .- mld_extrema(reference).minimum,
                    :balance, (-20, 20), "m")]))

monthly_mld_maps(o) = Ext.cached(() -> [Ext.regrid_to_map(o, abs.(m)) for m in Ext.monthly_means(o, "surface", "mlotst")],
                                 o, :monthly_mld_maps)

observed_monthly_mld = Ext.boyer_montegut_mixed_layer_depth()

month_names = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

month_directory = mkpath(joinpath(figure_directory, "mld_months"))

for m in 1:12
    figure = Ext.omip_figure(3, 2) do figure
        plot = nothing
        for (c, o) in enumerate(outputs)
            plot = Ext.global_map!(Ext.panel(figure, 1, c), monthly_mld_maps(o)[m]; title = string(o.label, ": ", month_names[m], " MLD"),
                                   colormap = Ext.mixed_layer_colormap, colorrange = (0, 400))
        end
        Ext.row_colorbar!(figure, 1, 3, plot, "m")

        isnothing(observed_monthly_mld) ||
            Ext.global_map!(Ext.panel(figure, 2, 1), observed_monthly_mld[m]; title = "de Boyer Montégut et al. (2004): " * month_names[m],
                            colormap = Ext.mixed_layer_colormap, colorrange = (0, 400))

        for (c, o) in enumerate(perturbed)
            plot = Ext.global_map!(Ext.panel(figure, 2, c + 1), monthly_mld_maps(o)[m] .- monthly_mld_maps(reference)[m];
                                   title = string(o.label, " − control"), colormap = :balance, colorrange = (-150, 150))
        end
        Ext.row_colorbar!(figure, 2, 3, plot, "m")
    end
    save(joinpath(month_directory, @sprintf("mld_%02d.png", m)), figure; px_per_unit = 1)
end
@info "Saved the monthly MLD maps"

# Regions on the 1ᵒ map grid: (title, latitude range, longitude range or nothing)
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


"""Zonal and regional means of every surface MLD record of the whole run."""
mld_series(o) = Ext.cached(o, :mld_series) do
    fts = Ext.timeseries(o, "surface", "mlotst")
    Nt = length(fts.times)
    zonal = fill(NaN, Nt, length(latitudes))
    regional = fill(NaN, Nt, length(regions))
    for n in 1:Nt
        map = Ext.regrid_to_map(o, abs.(Ext.record_array(fts, n)))
        zonal[n, :] .= [let row = filter(isfinite, view(map, :, j)); isempty(row) ? NaN : mean(row) end
                        for j in eachindex(latitudes)]
        regional[n, :] .= [regional_mean(map, region) for region in regions]
    end
    (; years = fts.times ./ 365days, zonal, regional)
end

save_figure("mld_hovmoller",
    Ext.omip_figure(length(outputs), 1; column_width = 640, row_height = 420) do figure
        control = mld_series(reference)
        axis = Axis(figure[1, 1]; title = reference.label * ": zonal-mean MLD", xlabel = "Year", ylabel = "Latitude")
        plot = heatmap!(axis, control.years, latitudes, log10.(control.zonal); colormap = Ext.mixed_layer_colormap,
                        colorrange = (1, 3))
        Colorbar(figure[2, 1], plot; label = "log₁₀ MLD (m)", vertical = false, flipaxis = false, width = Relative(0.7))
        for (c, o) in enumerate(perturbed)
            series = mld_series(o)
            Nt = min(length(series.years), length(control.years))
            ratio = log2.(series.zonal[1:Nt, :] ./ control.zonal[1:Nt, :])
            axis = Axis(figure[1, c + 1]; title = o.label * ": log₂(MLD / control MLD)", xlabel = "Year", ylabel = "Latitude")
            plot = heatmap!(axis, series.years[1:Nt], latitudes, ratio; colormap = :balance, colorrange = (-1, 1))
            Colorbar(figure[2, c + 1], plot; label = "log₂ ratio", vertical = false, flipaxis = false, width = Relative(0.7))
        end
    end)

observed_regional_climatology = isnothing(observed_monthly_mld) ? nothing :
    [[regional_mean(observed_monthly_mld[m], region) for m in 1:12] for region in regions]

save_figure("mld_regional_seasonal_cycle",
    Ext.omip_figure(length(regions), 1; column_width = 420, row_height = 380) do figure
        axes = map(enumerate(regions)) do (r, region)
            axis = Ext.climatology_axis(figure[1, r]; title = region[1], ylabel = "MLD (m)")
            isnothing(observed_regional_climatology) ||
                lines!(axis, 1:12, observed_regional_climatology[r]; Ext.observation_line..., label = "de Boyer Montégut et al. (2004)")
            for (n, o) in enumerate(outputs)
                Ext.run_lines!(axis, 1:12, [regional_mean(monthly_mld_maps(o)[m], region) for m in 1:12], n, o.label)
            end
            axis
        end
        Legend(figure[2, 1:length(regions)], axes[1]; orientation = :horizontal, framevisible = false, merge = true)
    end)

save_figure("mld_regional_series",
    Ext.omip_figure(length(regions), 1; column_width = 420, row_height = 380) do figure
        axes = map(enumerate(regions)) do (r, region)
            axis = Ext.time_axis(figure[1, r]; title = region[1], ylabel = "MLD (m)")
            for (n, o) in enumerate(outputs)
                series = mld_series(o)
                Ext.run_lines!(axis, series.years, series.regional[:, r], n, o.label)
            end
            vspan!(axis, first_year, last_year; color = (:gray, 0.15))
            axis
        end
        Legend(figure[2, 1:length(regions)], axes[1]; orientation = :horizontal, framevisible = false)
    end)

#####
##### Summary
#####

function band_mean(section, z, (φ₁, φ₂), depth)
    k = argmin(abs.(z .+ depth))
    rows = findall(φ -> φ₁ ≤ φ ≤ φ₂, latitudes)
    values = filter(isfinite, section[rows, k])
    return isempty(values) ? NaN : mean(values)
end

open(joinpath(figure_directory, "summary.csv"), "w") do io
    columns = ["run", "max_slope",
               ("$(name) $(statistic) MLD (m)" for (name, _, _) in regions for statistic in ("annual-mean", "max-month", "min-month"))...,
               "Southern Ocean slope at 1000 m, 60S-45S (1e-3)", "northern slope at 1000 m, 40N-60N (1e-3)",
               "min GM psi*, 70S-30S (Sv)", "upper-1000 m mean tapering factor, 60S-40S", "upper-1000 m mean tapering factor, 20S-20N"]
    println(io, join(replace.(columns, ", " => " "), ","))

    for (o, s) in zip(outputs, max_slopes)
        climatologies = [[regional_mean(monthly_mld_maps(o)[m], region) for m in 1:12] for region in regions]
        mld = [f(c) for c in climatologies for f in (mean, maximum, minimum)]
        z = Ext.depths(o)
        slope = 1e3 .* zonal_isopycnal_slope(o)
        φψ, zψ, ψ, _ = global_streamfunction(o, "vvolgm")
        southern = findall(φ -> -70 ≤ φ ≤ -30, φψ)
        upper = findall(>(-1000), z)
        taper = zonal_tapering(o)
        upper_taper(φ_range) = mean(filter(isfinite, taper[findall(φ -> φ_range[1] ≤ φ ≤ φ_range[2], latitudes), upper]))
        values = [o.label, s, mld...,
                  band_mean(slope, z, (-60, -45), 1000), band_mean(slope, z, (40, 60), 1000),
                  minimum(ψ[southern, :]), upper_taper((-60, -40)), upper_taper((-20, 20))]
        println(io, join((v isa Number ? @sprintf("%.4g", v) : v for v in values), ","))
    end

    if !isnothing(observed_regional_climatology)
        mld = [f(c) for c in observed_regional_climatology for f in (mean, maximum, minimum)]
        println(io, join(["de Boyer Montégut et al. (2004)", "", (@sprintf("%.4g", v) for v in mld)..., "", "", "", "", ""], ","))
    end
end
@info "Saved summary.csv"
