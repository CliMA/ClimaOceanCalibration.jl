# Figures comparing the one-degree OMIP runs of run_catke_diffusivity.jl: the CATKE diffusivities, the mixed-layer
# depth against the de Boyer Montégut et al. (2004) climatology through the seasonal cycle and region by region, the
# upper-ocean temperature against the World Ocean Atlas, and the equatorial Pacific.

using ArgParse
using ClimaOcean
using ClimaOcean: OMIPOutput, convert_to_teos10!
using CairoMakie
using GeoMakie
using NumericalEarth: Metadatum, WOAMonthly
using NumericalEarth.DataWrangling: all_dates
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models: buoyancy_operation
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Statistics
using Printf

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--runs-directory"
        arg_type = String
        default = expanduser("~/catke_diffusivity_sensitivity")
    "--figure-directory"
        arg_type = String
        default = expanduser("~/catke_diffusivity_sensitivity/figures")
    "--diffusivity-scales"
        arg_type = Float64
        nargs = '+'
        default = [0.5, 1.0, 2.0]
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
diffusivity_scales = arguments["diffusivity-scales"]
first_year, last_year = arguments["first-year"], arguments["last-year"]

const Ext = Base.get_extension(ClimaOcean, :ClimaOceanMakieExt)

run_name(scale) = "catke_" * string(scale)
run_label(scale) = scale == 1 ? "control" : @sprintf("κ × %g", scale)

averaging_window = (first_year * 365days + 1, last_year * 365days + 1)
outputs = [OMIPOutput(joinpath(runs_directory, run_name(s)); prefix = run_name(s), label = run_label(s), averaging_window)
           for s in diffusivity_scales]
reference = outputs[findfirst(==(1), diffusivity_scales)]
perturbed = filter(o -> o !== reference, outputs)
run_number(o) = findfirst(==(o), outputs)

function save_figure(name, figure; px_per_unit = 1.5)
    save(joinpath(figure_directory, name * ".png"), figure; px_per_unit)
    @info "Saved $name.png"
end

latitudes = Ext.map_latitudes
longitudes = Ext.map_longitudes
month_names = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
observation_title = "de Boyer Montégut et al. (2004)"
atlas_title = "World Ocean Atlas"

#####
##### Rows of panels sharing one colorbar; `nothing` leaves a column empty
#####

panel_row(panels, levels, colormap, label; xlabel = "Latitude", depth_limit = 1000, overlay_levels = levels) =
    (; panels, levels, colormap, label, xlabel, depth_limit, overlay_levels)

section(title, x, z, data; overlay = nothing, lines = ()) = (; title, x, z, data, overlay, lines)

"""Rows of depth sections. Each panel may carry `lines`, tuples of `(x, depth, color)` drawn over the section."""
function sections_figure(rows; column_width = 560, row_height = 330)
    columns = maximum(length(row.panels) for row in rows)
    return Ext.omip_figure(columns, length(rows); column_width, row_height) do figure
        for (r, row) in enumerate(rows)
            plot = nothing
            for (c, p) in enumerate(row.panels)
                isnothing(p) && continue
                plot = Ext.section_panel!(Ext.panel(figure, r, c), p.x, p.z, p.data; p.title, row.levels, row.colormap,
                                          p.overlay, row.overlay_levels, row.xlabel, row.depth_limit)
                for (x, depth, color) in p.lines
                    lines!(current_axis(), x, - depth; color, linewidth = 2.5)
                end
            end
            Ext.row_colorbar!(figure, r, columns, plot, row.label)
        end
    end
end

map_row(panels, colormap, colorrange, label) = (; panels, colormap, colorrange, label)

function maps_figure(rows)
    columns = maximum(length(row.panels) for row in rows)
    return Ext.omip_figure(columns, length(rows)) do figure
        for (r, row) in enumerate(rows)
            plot = nothing
            for (c, p) in enumerate(row.panels)
                isnothing(p) && continue
                title, data = p
                plot = Ext.global_map!(Ext.panel(figure, r, c), data; title, row.colormap, row.colorrange)
            end
            Ext.row_colorbar!(figure, r, columns, plot, row.label)
        end
    end
end

run_panels(title, data) = [(string(o.label, ": ", title), data(o)) for o in outputs]
difference_panels(title, data) = [o === reference ? nothing : (string(o.label, " − control: ", title), data(o) .- data(reference))
                                  for o in outputs]

#####
##### CATKE diffusivities and turbulent kinetic energy
#####

interface_depths(o) = Ext.cached(() -> collect(znodes(Ext.model_grid(o), Face()))[2:end-1], o, :interface_depths)
interface_mask(o) = Ext.cached(() -> (mask = Ext.ocean_mask(o); mask[:, :, 2:end] .& mask[:, :, 1:end-1]), o, :interface_mask)

zonal_diffusivity(o, name) = Ext.cached(o, Symbol(:zonal_, name)) do
    Ext.zonal_mean(o, Ext.time_mean(o, "diffusivities", name)[:, :, 2:end-1], interface_mask(o))
end

zonal_turbulent_kinetic_energy(o) = Ext.cached(() -> Ext.zonal_mean(o, Ext.time_mean(o, "fields", "tke")), o, :zonal_tke)

logarithm(f, x) = x > 0 ? f(x) : NaN
log_ratio(a, b) = logarithm.(log2, a ./ b)

diffusivity_rows(name, symbol) =
    [panel_row([section("$(o.label): log₁₀ zonal-mean $symbol", latitudes, interface_depths(o), logarithm.(log10, zonal_diffusivity(o, name)))
                for o in outputs], range(-6, 0, length = 13), :viridis, "log₁₀ $symbol (m² s⁻¹)"; depth_limit = 500),
     panel_row([o === reference ? nothing :
                section("$(o.label): log₂($symbol / control $symbol)", latitudes, interface_depths(o),
                        log_ratio(zonal_diffusivity(o, name), zonal_diffusivity(reference, name))) for o in outputs],
               range(-2, 2, length = 17), :balance, "log₂ ratio"; depth_limit = 500)]

save_figure("catke_diffusivities",
    sections_figure([diffusivity_rows("difvho", "κc")...,
                     diffusivity_rows("difvmo", "κu")...,
                     panel_row([o === reference ? nothing :
                                section("$(o.label): log₂(e / control e)", latitudes, Ext.depths(o),
                                        log_ratio(zonal_turbulent_kinetic_energy(o), zonal_turbulent_kinetic_energy(reference)))
                                for o in outputs], range(-2, 2, length = 17), :balance, "log₂ ratio"; depth_limit = 500)]))

#####
##### Mixed-layer depth, compared where the climatology has data in every month
#####

monthly_mld(o) = Ext.cached(o, :monthly_mld) do
    [Ext.regrid_to_map(o, m) for m in Ext.monthly_means(o, "mixed_layer", "mlotst_bracketing")]
end

observed_everywhere = let observed = Ext.boyer_montegut_mixed_layer_depth()
    compared = reduce(.&, isfinite.(m) for m in observed) .& isfinite.(monthly_mld(reference)[1])
    [ifelse.(compared, m, NaN) for m in observed]
end

compared_mld(o) = [ifelse.(isfinite.(observed_everywhere[1]), m, NaN) for m in monthly_mld(o)]
extremum_map(maps, f) = f(cat(maps...; dims = 3); dims = 3)[:, :, 1]
winter_maximum(maps) = extremum_map(maps, maximum)
summer_minimum(maps) = extremum_map(maps, minimum)
bias_panels(title, data) = [nothing, ((string(o.label, " − observed: ", title), data(compared_mld(o)) .- data(observed_everywhere))
                                      for o in outputs)...]

save_figure("mld_extrema_maps",
    maps_figure([map_row([(observation_title * ": winter maximum MLD", winter_maximum(observed_everywhere)),
                          run_panels("winter maximum MLD", o -> winter_maximum(monthly_mld(o)))...], Ext.mixed_layer_colormap, (0, 600), "m"),
                 map_row(bias_panels("winter maximum MLD", winter_maximum), :balance, (-200, 200), "m"),
                 map_row([(observation_title * ": summer minimum MLD", summer_minimum(observed_everywhere)),
                          run_panels("summer minimum MLD", o -> summer_minimum(monthly_mld(o)))...], Ext.mixed_layer_colormap, (0, 80), "m"),
                 map_row(bias_panels("summer minimum MLD", summer_minimum), :balance, (-40, 40), "m")]))

# The bias of the control and the response to the diffusivities, both as ratios
response_row(title, extremum) =
    map_row([("log₂(control / observed): " * title, log_ratio(extremum(compared_mld(reference)), extremum(observed_everywhere))),
             (("log₂($(o.label) / control): " * title, log_ratio(extremum(monthly_mld(o)), extremum(monthly_mld(reference))))
              for o in perturbed)...], :balance, (-1, 1), "log₂ ratio")

save_figure("mld_bias_and_response",
    maps_figure([response_row("winter maximum MLD", winter_maximum), response_row("summer minimum MLD", summer_minimum)]))

month_directory = mkpath(joinpath(figure_directory, "mld_months"))

for m in 1:12
    figure = maps_figure([map_row([(observation_title * ": " * month_names[m] * " MLD", observed_everywhere[m]),
                                   run_panels(month_names[m] * " MLD", o -> monthly_mld(o)[m])...],
                                  Ext.mixed_layer_colormap, (0, 300), "m"),
                          map_row(bias_panels(month_names[m] * " MLD", maps -> maps[m]), :balance, (-100, 100), "m")])
    save(joinpath(month_directory, @sprintf("mld_%02d.png", m)), figure; px_per_unit = 1)
end
@info "Saved the monthly MLD maps"

# Regions on the 1ᵒ map grid: (title, latitude range, longitude range from west to east or nothing)
bands = [("tropics, 20ᵒS–20ᵒN", (-20, 20), nothing),
         ("northern extratropics, 30ᵒN–60ᵒN", (30, 60), nothing),
         ("southern extratropics, 60ᵒS–30ᵒS", (-60, -30), nothing),
         ("Southern Ocean, 65ᵒS–50ᵒS", (-65, -50), nothing),
         ("subpolar North Atlantic, 50ᵒN–65ᵒN, 60ᵒW–10ᵒW", (50, 65), (300, 350))]

patches = [("Labrador Sea", (52, 65), (298, 315)),
           ("Nordic Seas", (65, 78), (345, 15)),
           ("North Atlantic subtropical gyre", (20, 40), (285, 340)),
           ("tropical Atlantic", (-10, 10), (320, 360)),
           ("Kuroshio extension", (30, 42), (140, 180)),
           ("subpolar North Pacific", (42, 58), (160, 220)),
           ("North Pacific subtropical gyre", (15, 30), (150, 230)),
           ("South Pacific subtropical gyre", (-35, -15), (180, 260)),
           ("western Pacific warm pool", (-10, 10), (140, 170)),
           ("central equatorial Pacific", (-5, 5), (190, 240)),
           ("eastern Pacific cold tongue", (-5, 5), (240, 275)),
           ("Arabian Sea", (8, 22), (55, 72)),
           ("tropical Indian Ocean", (-15, 5), (50, 95)),
           ("Southern Ocean, Atlantic sector", (-60, -45), (300, 20)),
           ("Southern Ocean, Indian sector", (-60, -45), (20, 150)),
           ("Southern Ocean, Pacific sector", (-60, -45), (150, 290))]

in_longitudes(λ, ::Nothing) = true
in_longitudes(λ, (west, east)) = mod(λ - west, 360) ≤ mod(east - west, 360)

function regional_mean(map, (_, (south, north), longitude_range))
    total = weight = 0.0
    for j in eachindex(latitudes), i in eachindex(longitudes)
        south ≤ latitudes[j] ≤ north || continue
        in_longitudes(longitudes[i], longitude_range) || continue
        isfinite(map[i, j]) || continue
        total += map[i, j] * cosd(latitudes[j])
        weight += cosd(latitudes[j])
    end
    return total / weight
end

regional_climatology(maps, region) = [regional_mean(maps[m], region) for m in 1:12]

function seasonal_cycle_figure(regions, columns)
    rows = cld(length(regions), columns)
    return Ext.omip_figure(columns, rows; column_width = 400, row_height = 300) do figure
        axes = map(enumerate(regions)) do (n, region)
            axis = Ext.climatology_axis(figure[cld(n, columns), mod1(n, columns)]; title = region[1], ylabel = "MLD (m)")
            axis.yreversed = true
            lines!(axis, 1:12, regional_climatology(observed_everywhere, region); Ext.observation_line..., label = observation_title)
            for o in outputs
                Ext.run_lines!(axis, 1:12, regional_climatology(compared_mld(o), region), run_number(o), o.label)
            end
            axis
        end
        Legend(figure[rows + 1, 1:columns], axes[1]; orientation = :horizontal, framevisible = false)
    end
end

save_figure("mld_seasonal_cycle_bands", seasonal_cycle_figure(bands, 5))
save_figure("mld_seasonal_cycle_patches", seasonal_cycle_figure(patches, 4))

zonal_mean_map(map) = [Ext.finite_mean(view(map, :, j)) for j in eachindex(latitudes)]
latitude_month(maps) = permutedims(hcat((zonal_mean_map(m) for m in maps)...))

save_figure("mld_latitude_month",
    Ext.omip_figure(length(outputs) + 1, 2; column_width = 420, row_height = 420) do figure
        observed = latitude_month(observed_everywhere)
        panels = [(observation_title, observed), ((o.label, latitude_month(compared_mld(o))) for o in outputs)...]
        plot = nothing
        for (c, (title, data)) in enumerate(panels)
            axis = Ext.climatology_axis(Ext.panel(figure, 1, c); title = title * ": zonal-mean MLD", ylabel = "Latitude")
            plot = heatmap!(axis, 1:12, latitudes, logarithm.(log10, data); colormap = Ext.mixed_layer_colormap, colorrange = (1, 2.7))
            ylims!(axis, -75, 75)
        end
        Ext.row_colorbar!(figure, 1, length(panels), plot, "log₁₀ MLD (m)")
        for (c, (title, data)) in enumerate(panels[2:end])
            axis = Ext.climatology_axis(Ext.panel(figure, 2, c + 1); title = "log₂($title / observed)", ylabel = "Latitude")
            plot = heatmap!(axis, 1:12, latitudes, log_ratio(data, observed); colormap = :balance, colorrange = (-1, 1))
            ylims!(axis, -75, 75)
        end
        Ext.row_colorbar!(figure, 2, length(panels), plot, "log₂ ratio")
    end)

save_figure("mld_zonal_extrema",
    Ext.omip_figure(2, 1; column_width = 640, row_height = 380) do figure
        axes = map(enumerate((("summer minimum", summer_minimum), ("winter maximum", winter_maximum)))) do (c, (title, extremum))
            axis = Axis(figure[1, c]; title = "zonal-mean $title MLD", xlabel = "Latitude", ylabel = "MLD (m)")
            lines!(axis, latitudes, zonal_mean_map(extremum(observed_everywhere)); Ext.observation_line..., label = observation_title)
            for o in outputs
                Ext.run_lines!(axis, latitudes, zonal_mean_map(extremum(compared_mld(o))), run_number(o), o.label)
            end
            xlims!(axis, -78, 78)
            axis
        end
        Legend(figure[2, 1:2], axes[1]; orientation = :horizontal, framevisible = false)
    end)

"""Zonal and band means of every 5-day mixed-layer depth record of the whole run."""
mld_series(o) = Ext.cached(o, :mld_series) do
    fts = Ext.timeseries(o, "mixed_layer", "mlotst_bracketing")
    Nt = length(fts.times)
    zonal = fill(NaN, Nt, length(latitudes))
    regional = fill(NaN, Nt, length(bands))
    for n in 1:Nt
        map = Ext.regrid_to_map(o, Ext.record_array(fts, n))
        zonal[n, :] .= zonal_mean_map(map)
        regional[n, :] .= [regional_mean(map, band) for band in bands]
    end
    (; years = fts.times ./ 365days, zonal, regional)
end

save_figure("mld_hovmoller",
    Ext.omip_figure(length(outputs), 1; column_width = 640, row_height = 420) do figure
        control = mld_series(reference)
        axis = Axis(figure[1, 1]; title = "control: zonal-mean MLD", xlabel = "Year", ylabel = "Latitude")
        plot = heatmap!(axis, control.years, latitudes, log10.(control.zonal); colormap = Ext.mixed_layer_colormap, colorrange = (1, 3))
        Colorbar(figure[2, 1], plot; label = "log₁₀ MLD (m)", vertical = false, flipaxis = false, width = Relative(0.7))
        for (c, o) in enumerate(perturbed)
            series = mld_series(o)
            Nt = min(length(series.years), length(control.years))
            axis = Axis(figure[1, c + 1]; title = "log₂($(o.label) MLD / control MLD)", xlabel = "Year", ylabel = "Latitude")
            plot = heatmap!(axis, series.years[1:Nt], latitudes, log2.(series.zonal[1:Nt, :] ./ control.zonal[1:Nt, :]);
                            colormap = :balance, colorrange = (-1, 1))
            Colorbar(figure[2, c + 1], plot; label = "log₂ ratio", vertical = false, flipaxis = false, width = Relative(0.7))
        end
    end)

save_figure("mld_regional_series",
    Ext.omip_figure(length(bands), 1; column_width = 420, row_height = 380) do figure
        axes = map(enumerate(bands)) do (r, band)
            axis = Ext.time_axis(figure[1, r]; title = band[1], ylabel = "MLD (m)")
            for o in outputs
                series = mld_series(o)
                Ext.run_lines!(axis, series.years, series.regional[:, r], run_number(o), o.label)
            end
            vspan!(axis, first_year, last_year; color = (:gray, 0.15))
            axis
        end
        Legend(figure[2, 1:length(bands)], axes[1]; orientation = :horizontal, framevisible = false)
    end)

open(joinpath(figure_directory, "mld_regions.csv"), "w") do io
    statistics = (("annual mean", mean), ("deepest month", maximum), ("shallowest month", minimum))
    println(io, join(["region", "statistic", observation_title, (o.label for o in outputs)...], ","))
    for region in vcat(bands, patches), (statistic, f) in statistics
        values = [f(regional_climatology(maps, region)) for maps in (observed_everywhere, compared_mld.(outputs)...)]
        println(io, join([replace(region[1], ", " => " "), statistic, (@sprintf("%.4g", v) for v in values)...], ","))
    end
end
@info "Saved mld_regions.csv"

#####
##### Temperature against the World Ocean Atlas, and the surface
#####

zonal_temperature_bias(o) = Ext.zonal_sections(o).temperature .- Ext.zonal_sections(o).atlas_temperature
zonal_section(title, o, data) = section(title, latitudes, Ext.depths(o), data)

save_figure("zonal_temperature",
    sections_figure([panel_row([zonal_section("$(o.label): zonal-mean T − WOA", o, zonal_temperature_bias(o)) for o in outputs],
                               range(-3, 3, length = 13), :balance, "ᵒC"),
                     panel_row([o === reference ? nothing :
                                zonal_section("$(o.label) − control: zonal-mean T", o,
                                              Ext.zonal_sections(o).temperature .- Ext.zonal_sections(reference).temperature)
                                for o in outputs], range(-1.5, 1.5, length = 13), :balance, "ᵒC"),
                     panel_row([o === reference ? nothing :
                                zonal_section("$(o.label) − control: zonal-mean S", o,
                                              Ext.zonal_sections(o).salinity .- Ext.zonal_sections(reference).salinity)
                                for o in outputs], range(-0.3, 0.3, length = 13), :balance, "g kg⁻¹")]))

save_figure("sea_surface",
    maps_figure([map_row(run_panels("SST − WOA", Ext.sea_surface_temperature_bias_map), :balance, (-3, 3), "ᵒC"),
                 map_row(difference_panels("SST", Ext.sea_surface_temperature_bias_map), :balance, (-1, 1), "ᵒC"),
                 map_row(difference_panels("net heat flux into the ocean", Ext.net_heat_flux_map), :balance, (-30, 30), "W m⁻²")]))

save_figure("mean_profile_drift",
    Ext.omip_figure(2, 1; column_width = 480, row_height = 520) do figure
        panels = (("horizontal-mean temperature drift", "to_h", "ᵒC"), ("horizontal-mean salinity drift", "so_h", "g kg⁻¹"))
        axes = map(enumerate(panels)) do (c, (title, name, unit))
            axis = Axis(figure[1, c]; title, xlabel = "years $(first_year)–$(last_year) − first 5 days ($unit)", ylabel = "Depth (m)")
            for o in outputs
                Ext.run_lines!(axis, Ext.mean_profile(o, name) .- Ext.profile_series(o, name)[1, :], Ext.depths(o), run_number(o), o.label)
            end
            vlines!(axis, 0; color = (:black, 0.4), linewidth = 1)
            ylims!(axis, -1500, 0)
            axis
        end
        Legend(figure[2, 1:2], axes[1]; orientation = :horizontal, framevisible = false)
    end)

#####
##### Profiles averaged over boxes of the model grid
#####

"""The mean of the three-dimensional `data` of `o` over the wet cells of each `(longitude range, latitude range)` box."""
function box_profiles(o, data, boxes)
    λ, φ, mask = Ext.center_longitudes(o), Ext.center_latitudes(o), Ext.ocean_mask(o)
    profiles = fill(NaN, length(boxes), size(data, 3))
    for (n, (longitude_range, (south, north))) in enumerate(boxes)
        cells = findall(c -> south ≤ φ[c] ≤ north && in_longitudes(λ[c], longitude_range), CartesianIndices(λ))
        for k in axes(data, 3)
            values = [data[c, k] for c in cells if mask[c, k] && isfinite(data[c, k])]
            isempty(values) || (profiles[n, k] = mean(values))
        end
    end
    return profiles
end

"""The mean of a map on the 1ᵒ grid over each box."""
box_means(map, boxes) = [regional_mean(map, ("", latitude_range, longitude_range)) for (longitude_range, latitude_range) in boxes]

world_ocean_atlas_monthly = let grid = Ext.model_grid(reference), mask = Ext.ocean_mask(reference)
    map(all_dates(WOAMonthly(), :temperature)) do date
        T = CenterField(grid)
        S = CenterField(grid)
        set!(T, Metadatum(:temperature; dataset = WOAMonthly(), date))
        set!(S, Metadatum(:salinity; dataset = WOAMonthly(), date))
        convert_to_teos10!(T, S)
        Ext.masked(Array(interior(T)), mask)
    end
end

world_ocean_atlas_buoyancy = let grid = Ext.model_grid(reference), atlas = Ext.world_ocean_atlas(reference)
    T = CenterField(grid)
    S = CenterField(grid)
    set!(T, ifelse.(isfinite.(atlas.T), atlas.T, 0))
    set!(S, ifelse.(isfinite.(atlas.S), atlas.S, 35))
    buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState(reference_density = Ext.reference_density))
    b = Field(buoyancy_operation(buoyancy, grid, (; T, S)))
    Ext.masked(Array(interior(compute!(b))), Ext.ocean_mask(reference))
end

annual_mean(maps) = sum(maps) ./ length(maps)

#####
##### The equatorial Pacific, averaged over 2ᵒS–2ᵒN
#####

equator = (-2, 2)
section_longitudes = 130.5:1:279.5
section_boxes = [((λ - 0.5, λ + 0.5), equator) for λ in section_longitudes]
profile_longitudes = 140:10:270
profile_boxes = [((λ - 2, λ + 2), equator) for λ in profile_longitudes]

equatorial_section(o, group, name) = Ext.cached(() -> box_profiles(o, Ext.time_mean(o, group, name), section_boxes),
                                                o, Symbol(:equatorial_section_, name))

"""The depth at which each profile of `sections[longitude, level]` first drops below `value`, from the surface down."""
function isotherm_depth(sections, z, value)
    map(axes(sections, 1)) do n
        for k in length(z)-1:-1:1
            above, below = sections[n, k+1], sections[n, k]
            isfinite(above) && isfinite(below) && above ≥ value > below || continue
            return - (z[k+1] + (z[k] - z[k+1]) * (above - value) / (above - below))
        end
        NaN
    end
end

let z = Ext.depths(reference), atlas = Ext.world_ocean_atlas(reference)
    atlas_section = box_profiles(reference, atlas.T, section_boxes)
    observed_mld = box_means(annual_mean(observed_everywhere), section_boxes)
    run_mld(o) = box_means(annual_mean(monthly_mld(o)), section_boxes)
    temperature(o) = equatorial_section(o, "fields", "to")
    undercurrent(o) = Ext.equatorial_sections(o)
    mld_line(depth) = (section_longitudes, depth, :white)
    kw = (; xlabel = "Longitude (ᵒE)", depth_limit = 400)

    save_figure("equatorial_pacific_sections",
        sections_figure([panel_row([section("$atlas_title: T, 2ᵒS–2ᵒN", section_longitudes, z, atlas_section;
                                            overlay = atlas_section, lines = (mld_line(observed_mld),)),
                                    (section("$(o.label): T, 2ᵒS–2ᵒN", section_longitudes, z, temperature(o);
                                             overlay = temperature(o), lines = (mld_line(run_mld(o)),)) for o in outputs)...],
                                   8:1:30, :thermal, "ᵒC, 20 ᵒC dashed, mixed-layer depth solid"; overlay_levels = [20], kw...),
                         panel_row([nothing, (section("$(o.label): T − WOA", section_longitudes, z, temperature(o) .- atlas_section)
                                              for o in outputs)...], range(-4, 4, length = 17), :balance, "ᵒC"; kw...),
                         panel_row([nothing, (o === reference ? nothing :
                                              section("$(o.label) − control: T", section_longitudes, z, temperature(o) .- temperature(reference))
                                              for o in outputs)...], range(-2, 2, length = 17), :balance, "ᵒC"; kw...),
                         panel_row([nothing, (section("$(o.label): eastward velocity on the equator", undercurrent(o).longitudes,
                                                      undercurrent(o).depths, 100 .* undercurrent(o).along_equator) for o in outputs)...],
                                   -100:10:100, :balance, "cm s⁻¹"; kw...)]))

    save_figure("equatorial_pacific_lines",
        Ext.omip_figure(4, 1; column_width = 420, row_height = 380) do figure
            kw = (; xlabel = "Longitude (ᵒE)")
            surface = Axis(figure[1, 1]; title = "sea surface temperature", ylabel = "ᵒC", kw...)
            mixed_layer = Axis(figure[1, 2]; title = "mixed-layer depth", ylabel = "m", yreversed = true, kw...)
            thermocline = Axis(figure[1, 3]; title = "depth of the 20 ᵒC isotherm", ylabel = "m", yreversed = true, kw...)
            current = Axis(figure[1, 4]; title = "fastest eastward velocity on the equator", ylabel = "cm s⁻¹", kw...)

            lines!(surface, section_longitudes, atlas_section[:, end]; Ext.observation_line..., label = "observed")
            lines!(mixed_layer, section_longitudes, observed_mld; Ext.observation_line...)
            lines!(thermocline, section_longitudes, isotherm_depth(atlas_section, z, 20); Ext.observation_line...)
            for o in outputs
                n = run_number(o)
                Ext.run_lines!(surface, section_longitudes, temperature(o)[:, end], n, o.label)
                Ext.run_lines!(mixed_layer, section_longitudes, run_mld(o), n, o.label)
                Ext.run_lines!(thermocline, section_longitudes, isotherm_depth(temperature(o), z, 20), n, o.label)
                Ext.run_lines!(current, undercurrent(o).longitudes,
                               [maximum(filter(isfinite, 100 .* row); init = NaN) for row in eachrow(undercurrent(o).along_equator)], n, o.label)
            end
            Legend(figure[2, 1:4], surface; orientation = :horizontal, framevisible = false)
        end)

    # Profiles every 10ᵒ of longitude, with a marker at the mixed-layer depth of each
    variables = (("temperature (ᵒC)", "to", atlas.T), ("salinity (g kg⁻¹)", "so", atlas.S), ("buoyancy (m s⁻²)", "bo", world_ocean_atlas_buoyancy))
    profile_mld(maps) = box_means(annual_mean(maps), profile_boxes)
    columns = [(atlas_title, nothing, profile_mld(observed_everywhere)), ((o.label, o, profile_mld(monthly_mld(o))) for o in outputs)...]
    colors = cgrad(:viridis, length(profile_longitudes); categorical = true)

    save_figure("equatorial_pacific_profiles",
        Ext.omip_figure(length(columns), length(variables); column_width = 380, row_height = 440) do figure
            for (r, (xlabel, name, atlas_data)) in enumerate(variables), (c, (title, o, mld)) in enumerate(columns)
                profiles = box_profiles(reference, isnothing(o) ? atlas_data : Ext.time_mean(o, "fields", name), profile_boxes)
                axis = Axis(figure[r, c]; title, xlabel, ylabel = "Depth (m)")
                for n in eachindex(profile_longitudes)
                    lines!(axis, profiles[n, :], z; color = colors[n], linewidth = 2)
                    k = searchsortedfirst(z, - mld[n])
                    1 < k ≤ length(z) || continue
                    weight = (- mld[n] - z[k-1]) / (z[k] - z[k-1])
                    scatter!(axis, [profiles[n, k-1] + weight * (profiles[n, k] - profiles[n, k-1])], [- mld[n]];
                             color = colors[n], strokecolor = :black, strokewidth = 1, markersize = 11)
                end
                ylims!(axis, -300, 0)
            end
            Colorbar(figure[length(variables) + 1, 1:length(columns)]; colormap = colors, limits = (135, 275),
                     ticks = (140:20:260, ["140ᵒE", "160ᵒE", "180ᵒ", "160ᵒW", "140ᵒW", "120ᵒW", "100ᵒW"]),
                     label = "longitude of the profile; markers show the mixed-layer depth",
                     vertical = false, flipaxis = false, width = Relative(0.5))
        end)

    section_month_directory = mkpath(joinpath(figure_directory, "equatorial_pacific_months"))
    monthly_temperature = Dict(o => [box_profiles(o, T, section_boxes) for T in Ext.monthly_means(o, "fields", "to")] for o in outputs)
    for m in 1:12
        atlas_month = box_profiles(reference, world_ocean_atlas_monthly[m], section_boxes)
        figure = sections_figure([panel_row([section("$atlas_title: $(month_names[m]) T, 2ᵒS–2ᵒN", section_longitudes, z, atlas_month;
                                                     overlay = atlas_month, lines = (mld_line(box_means(observed_everywhere[m], section_boxes)),)),
                                             (section("$(o.label): $(month_names[m]) T", section_longitudes, z, monthly_temperature[o][m];
                                                      overlay = monthly_temperature[o][m],
                                                      lines = (mld_line(box_means(monthly_mld(o)[m], section_boxes)),)) for o in outputs)...],
                                            8:1:30, :thermal, "ᵒC, 20 ᵒC dashed, mixed-layer depth solid"; overlay_levels = [20], kw...)])
        save(joinpath(section_month_directory, @sprintf("section_%02d.png", m)), figure; px_per_unit = 1)
    end
    @info "Saved the monthly equatorial Pacific sections"
end

#####
##### The seasonal cycle of temperature and mixed-layer depth at sites of 4ᵒ × 4ᵒ
#####

# (title, longitude, latitude, depth of the panel)
sites = [("Ocean Station Papa, 50ᵒN 145ᵒW", 215, 50, 250),
         ("Kuroshio extension, 35ᵒN 150ᵒE", 150, 35, 400),
         ("Bermuda, 32ᵒN 64ᵒW", 296, 32, 400),
         ("Irminger Sea, 60ᵒN 38ᵒW", 322, 60, 800),
         ("Arabian Sea, 15ᵒN 62ᵒE", 62, 15, 200),
         ("equatorial Pacific, 0ᵒ 140ᵒW", 220, 0, 250),
         ("southeast Pacific, 55ᵒS 90ᵒW", 270, -55, 600),
         ("south Indian Ocean, 45ᵒS 90ᵒE", 90, -45, 600)]

save_figure("seasonal_cycle_sites",
    let z = Ext.depths(reference)
        rows = map(sites) do (title, λ, φ, depth_limit)
            box = [((λ - 2, λ + 2), (φ - 2, φ + 2))]
            climatology(o, monthly) = reduce(vcat, box_profiles(o, T, box) for T in monthly)
            site_mld(maps) = [only(box_means(maps[m], box)) for m in 1:12]
            observed_line = (1:12, site_mld(observed_everywhere), Ext.observation_color)
            atlas = climatology(reference, world_ocean_atlas_monthly)
            shallow = filter(isfinite, atlas[:, z .≥ - depth_limit])
            panels = [section("$atlas_title: $title", 1:12, z, atlas; lines = (observed_line,)),
                      (section("$(o.label): $title", 1:12, z, climatology(o, Ext.monthly_means(o, "fields", "to"));
                               lines = (observed_line, (1:12, site_mld(monthly_mld(o)), :white))) for o in outputs)...]
            panel_row(panels, range(floor(minimum(shallow)), ceil(maximum(shallow)), length = 15), :thermal,
                      "ᵒC; mixed-layer depth: observed gray, run white"; xlabel = "Month", depth_limit)
        end
        sections_figure(rows; column_width = 420, row_height = 300)
    end)
