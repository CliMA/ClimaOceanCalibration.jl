# plot_flux_comparison.jl
#
# Per-variable comparison figures: NumericalEarth JRA55×WOA prescribed-ocean
# fluxes vs the OMIP2 models' published climatologies.
#
# Consumes the processed 1°-regrid NetCDFs (`clim_r`) written by
#   fetch_omip2_fluxes.py        (OMIP2 models → omip_data/omip2_fluxes/)
#   postprocess_flux_climatology.jl  (our runs → <run>_run/climatology/)
# and treats every file uniformly: <var>_omip2_<LABEL>.nc.
#
# Usage:
#   julia +1.12.3 --project=<repo> plot_flux_comparison.jl [data_dir ...]
#
#   Default data dirs: <repo>/omip_data/omip2_fluxes plus any dirs passed as
#   arguments (e.g. the climatology/ dir of a finished run). Figures go to
#   FIG_DIR (default <repo>/omip_data/omip2_fluxes/figures).
#
# One figure per variable:
#   • annual-mean maps for every source (ours first, highlighted),
#   • ours − OMIP-multi-model-mean difference map,
#   • zonal means (annual) for all sources,
#   • seasonal cycle of the 60°S–60°N area-weighted mean.
#
# Additionally (VIDEOS=true, the default) one video per variable
# (fluxcomp_<var>_monthly.mp4): the same map layout animated through the 12
# climatological months, with a month cursor on the seasonal-cycle panel.
# Set VIDEOS=false to skip.
#
# Missing OMIP variables are derived where possible:
#   hfls ≈ Lv·evs                        (models publishing evs)
#   hfss = rsntds + rlntds − hfds − hfls (models publishing all four)

using CairoMakie
using NCDatasets
using Printf
using Statistics

const HERE = @__DIR__
const REPO = dirname(dirname(HERE))
const Lv = 2.5e6   # J/kg

data_dirs = String[joinpath(REPO, "omip_data", "omip2_fluxes")]
append!(data_dirs, abspath.(ARGS))
fig_dir = get(ENV, "FIG_DIR", joinpath(first(data_dirs), "figures"))
mkpath(fig_dir)

# ============================================
# Load all processed files: (var, label) → (12, nlat, nlon) on the 1° grid
# ============================================
const lat_r = collect(-89.5:1.0:89.5)
const lon_r = collect(0.5:1.0:359.5)

clim = Dict{Tuple{String, String}, Array{Float64, 3}}()   # (var, label) => (12, 180, 360)
for dir in data_dirs
    isdir(dir) || continue
    for f in sort(readdir(dir))
        m = match(r"^([a-z]+)_omip2_(.+)\.nc$", f)
        m === nothing && continue
        var, label = m.captures
        NCDataset(joinpath(dir, f)) do ds
            haskey(ds, "clim_r") || return
            A = coalesce.(ds["clim_r"][:, :, :], NaN)          # NCDatasets: (lon, lat, month); missing → NaN
            clim[(var, label)] = permutedims(Float64.(A), (3, 2, 1))   # → (month, lat, lon)
        end
    end
end
isempty(clim) && error("No processed <var>_omip2_<label>.nc files found in: $(join(data_dirs, ", "))")

labels_all = sort(unique(last.(keys(clim))))
ours(label)  = startswith(label, "NumericalEarth")
labels_ne    = filter(ours, labels_all)
labels_omip  = filter(!ours, labels_all)
@info "Sources: ours = $(labels_ne); OMIP2 = $(labels_omip)"

# Derived OMIP variables
for lab in labels_omip
    if !haskey(clim, ("hfls", lab)) && haskey(clim, ("evs", lab))
        clim[("hfls", lab)] = Lv .* clim[("evs", lab)]
    end
    if !haskey(clim, ("hfss", lab)) &&
       all(haskey(clim, (v, lab)) for v in ("rsntds", "rlntds", "hfds", "hfls"))
        clim[("hfss", lab)] = clim[("rsntds", lab)] .+ clim[("rlntds", lab)] .-
                              clim[("hfds", lab)] .- clim[("hfls", lab)]
    end
end

# ============================================
# Helpers
# ============================================
annual(c)     = dropdims(mapslices(x -> all(isnan, x) ? NaN : mean(filter(!isnan, x)), c; dims = 1); dims = 1)
nanmean(v)    = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
zonal_mean(a) = [nanmean(a[j, :]) for j in eachindex(lat_r)]

const w_lat = cosd.(lat_r)
function band_mean(a; band = 60)                    # area-weighted mean of a (lat, lon) map
    s = 0.0; W = 0.0
    for (j, φ) in enumerate(lat_r)
        abs(φ) > band && continue
        for i in eachindex(lon_r)
            isnan(a[j, i]) && continue
            s += w_lat[j] * a[j, i]; W += w_lat[j]
        end
    end
    return W > 0 ? s / W : NaN
end
seasonal_cycle(c; band = 60) = [band_mean(c[m, :, :]; band) for m in 1:12]

# Per-variable presentation: (title, units, scale, colormap, colorrange, diff_range)
mmday = 86400.0
const VARSPEC = Dict(
    "hfds"   => (title = "Net downward heat flux (hfds)",        units = "W/m²",   scale = 1.0,   cmap = :balance,  crange = (-200, 200), drange = (-60, 60)),
    "rsntds" => (title = "Net downward shortwave (rsntds)",      units = "W/m²",   scale = 1.0,   cmap = :thermal,  crange = (0, 280),    drange = (-40, 40)),
    "rlntds" => (title = "Net downward longwave (rlntds)",       units = "W/m²",   scale = 1.0,   cmap = :thermal,  crange = (-90, 0),    drange = (-25, 25)),
    "hfls"   => (title = "Upward latent heat flux (hfls)",       units = "W/m²",   scale = 1.0,   cmap = :thermal,  crange = (0, 250),    drange = (-50, 50)),
    "hfss"   => (title = "Upward sensible heat flux (hfss)",     units = "W/m²",   scale = 1.0,   cmap = :balance,  crange = (-40, 80),   drange = (-25, 25)),
    "evs"    => (title = "Evaporation (evs)",                    units = "mm/day", scale = mmday, cmap = :thermal,  crange = (0, 9),      drange = (-2, 2)),
    "prra"   => (title = "Rainfall onto ocean (prra)",           units = "mm/day", scale = mmday, cmap = :dense,    crange = (0, 12),     drange = (-3, 3)),
    "prsn"   => (title = "Snowfall onto ocean (prsn)",           units = "mm/day", scale = mmday, cmap = :dense,    crange = (0, 3),      drange = (-1, 1)),
    "wfo"    => (title = "Net water flux into ocean (wfo)",      units = "mm/day", scale = mmday, cmap = :balance,  crange = (-8, 8),     drange = (-3, 3)),
)
const VARORDER = ["hfls", "hfss", "evs", "prra", "prsn", "hfds", "rsntds", "rlntds", "wfo"]

const NE_COLORS = Dict("NumericalEarth-corrected" => :crimson, "NumericalEarth-ncar" => :darkorange)
omip_palette = Makie.wong_colors()

function map_panel!(fig, pos, a, spec; title, highlight = false)
    ax = Axis(fig[pos...]; title, titlefont = highlight ? :bold : :regular,
              titlecolor = highlight ? :crimson : :black,
              xlabel = "", ylabel = "", limits = ((0, 360), (-80, 90)),
              xticks = 0:90:360, yticks = -60:30:60)
    hm = heatmap!(ax, lon_r, lat_r, permutedims(a);
                  colormap = spec.cmap, colorrange = spec.crange, nan_color = (:gray70, 1))
    return hm
end

# ============================================
# Figures
# ============================================
for var in VARORDER
    spec = VARSPEC[var]
    sources = vcat([l for l in labels_ne if haskey(clim, (var, l))],
                   [l for l in labels_omip if haskey(clim, (var, l))])
    isempty(sources) && (@warn "no data at all for $var"; continue)
    have_ne = any(ours, sources)

    ann = Dict(l => spec.scale .* annual(clim[(var, l)]) for l in sources)

    omip_sources = filter(!ours, sources)
    mmm = isempty(omip_sources) ? nothing :
          [nanmean([ann[l][j, i] for l in omip_sources]) for j in eachindex(lat_r), i in eachindex(lon_r)]

    npanels = length(sources) + (mmm !== nothing ? 1 : 0) + (have_ne && mmm !== nothing ? 1 : 0)
    ncols = 3
    nrows_maps = ceil(Int, npanels / ncols)

    fig = Figure(size = (560 * ncols, 300 * nrows_maps + 360), fontsize = 15)
    Label(fig[0, 1:ncols], "$(spec.title) — annual mean, omip2 last-cycle climatology vs JRA55×WOA prescribed ocean";
          fontsize = 20, font = :bold, padding = (0, 0, 8, 0))

    hm = nothing
    panel = 0
    for l in sources
        panel += 1
        r, c = fldmod1(panel, ncols)
        hm = map_panel!(fig, (r, c), ann[l], spec; title = l, highlight = ours(l))
    end
    if mmm !== nothing
        panel += 1
        r, c = fldmod1(panel, ncols)
        map_panel!(fig, (r, c), mmm, spec; title = "OMIP2 multi-model mean ($(length(omip_sources)))")
        if have_ne
            panel += 1
            r, c = fldmod1(panel, ncols)
            diff = ann[first(filter(ours, sources))] .- mmm
            sub = GridLayout(fig[r, c])
            ax = Axis(sub[1, 1]; title = "$(first(filter(ours, sources))) − MMM", titlefont = :bold,
                      limits = ((0, 360), (-80, 90)), xticks = 0:90:360, yticks = -60:30:60)
            dhm = heatmap!(ax, lon_r, lat_r, permutedims(diff);
                           colormap = :balance, colorrange = spec.drange, nan_color = (:gray70, 1))
            Colorbar(sub[2, 1], dhm; label = "Δ $(spec.units)", vertical = false, flipaxis = false)
        end
    end
    hm !== nothing && Colorbar(fig[1:nrows_maps, ncols + 1], hm; label = spec.units)

    # Bottom row: zonal mean + seasonal cycle
    axz = Axis(fig[nrows_maps + 1, 1:2]; xlabel = "latitude", ylabel = spec.units,
               title = "Zonal mean (annual)", xticks = -80:20:80)
    axs = Axis(fig[nrows_maps + 1, 3]; xlabel = "month", ylabel = spec.units,
               title = "Seasonal cycle, 60°S–60°N mean", xticks = 1:12)
    for (k, l) in enumerate(omip_sources)
        color = omip_palette[mod1(k, length(omip_palette))]
        lines!(axz, lat_r, zonal_mean(ann[l]); color, alpha = 0.7, linewidth = 1.5, label = l)
        lines!(axs, 1:12, spec.scale .* seasonal_cycle(clim[(var, l)]); color, alpha = 0.7, linewidth = 1.5)
    end
    mmm !== nothing && lines!(axz, lat_r, zonal_mean(mmm); color = :black, linewidth = 2.5, label = "OMIP2 MMM")
    for l in filter(ours, sources)
        color = get(NE_COLORS, l, :crimson)
        lines!(axz, lat_r, zonal_mean(ann[l]); color, linewidth = 3, label = l)
        lines!(axs, 1:12, spec.scale .* seasonal_cycle(clim[(var, l)]); color, linewidth = 3)
    end
    axislegend(axz; position = :rb, nbanks = 2, framevisible = false, labelsize = 11)

    out = joinpath(fig_dir, "fluxcomp_$(var).png")
    save(out, fig)
    @info "saved $out"
end

# ============================================
# Monthly-climatology videos (same layout, animated through the 12 months)
# ============================================
const MONTHNAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

function monthly_video(var)
    spec = VARSPEC[var]
    sources = vcat([l for l in labels_ne if haskey(clim, (var, l))],
                   [l for l in labels_omip if haskey(clim, (var, l))])
    isempty(sources) && return
    have_ne = any(ours, sources)

    monthly = Dict(l => spec.scale .* clim[(var, l)] for l in sources)   # (12, lat, lon)

    omip_sources = filter(!ours, sources)
    mmm = isempty(omip_sources) ? nothing :
          [nanmean([monthly[l][m, j, i] for l in omip_sources])
           for m in 1:12, j in eachindex(lat_r), i in eachindex(lon_r)]

    # Seasonal amplitude exceeds the annual-mean spread — widen ranges a bit.
    crange = spec.crange[1] < 0 ? spec.crange .* 1.5 : (spec.crange[1], spec.crange[2] * 1.5)
    drange = spec.drange .* 1.5

    npanels = length(sources) + (mmm !== nothing ? 1 : 0) + (have_ne && mmm !== nothing ? 1 : 0)
    ncols = 3
    nrows_maps = ceil(Int, npanels / ncols)

    fig = Figure(size = (560 * ncols, 300 * nrows_maps + 340), fontsize = 15)
    imonth = Observable(1)
    Label(fig[0, 1:ncols],
          @lift("$(spec.title) — month: " * MONTHNAMES[$imonth] * " (omip2 climatology vs JRA55×WOA prescribed ocean)");
          fontsize = 20, font = :bold, padding = (0, 0, 8, 0))

    hm = nothing
    panel = 0
    for l in sources
        panel += 1
        r, c = fldmod1(panel, ncols)
        ax = Axis(fig[r, c]; title = l, titlefont = ours(l) ? :bold : :regular,
                  titlecolor = ours(l) ? :crimson : :black,
                  limits = ((0, 360), (-80, 90)), xticks = 0:90:360, yticks = -60:30:60)
        A = monthly[l]
        hm = heatmap!(ax, lon_r, lat_r, @lift(permutedims(@view A[$imonth, :, :]));
                      colormap = spec.cmap, colorrange = crange, nan_color = (:gray70, 1))
    end
    if mmm !== nothing
        panel += 1
        r, c = fldmod1(panel, ncols)
        ax = Axis(fig[r, c]; title = "OMIP2 multi-model mean ($(length(omip_sources)))",
                  limits = ((0, 360), (-80, 90)), xticks = 0:90:360, yticks = -60:30:60)
        heatmap!(ax, lon_r, lat_r, @lift(permutedims(@view mmm[$imonth, :, :]));
                 colormap = spec.cmap, colorrange = crange, nan_color = (:gray70, 1))
        if have_ne
            panel += 1
            r, c = fldmod1(panel, ncols)
            l1 = first(filter(ours, sources))
            sub = GridLayout(fig[r, c])
            ax = Axis(sub[1, 1]; title = "$l1 − MMM", titlefont = :bold,
                      limits = ((0, 360), (-80, 90)), xticks = 0:90:360, yticks = -60:30:60)
            A1 = monthly[l1]
            dhm = heatmap!(ax, lon_r, lat_r,
                           @lift(permutedims(@view(A1[$imonth, :, :]) .- @view(mmm[$imonth, :, :])));
                           colormap = :balance, colorrange = drange, nan_color = (:gray70, 1))
            Colorbar(sub[2, 1], dhm; label = "Δ $(spec.units)", vertical = false, flipaxis = false)
        end
    end
    hm !== nothing && Colorbar(fig[1:nrows_maps, ncols + 1], hm; label = spec.units)

    # Seasonal-cycle panel with a month cursor
    axs = Axis(fig[nrows_maps + 1, 1:ncols]; xlabel = "month", ylabel = spec.units,
               title = "Seasonal cycle, 60°S–60°N mean", xticks = 1:12)
    for (k, l) in enumerate(omip_sources)
        lines!(axs, 1:12, [band_mean(monthly[l][m, :, :]) for m in 1:12];
               color = omip_palette[mod1(k, length(omip_palette))], alpha = 0.7, linewidth = 1.5, label = l)
    end
    for l in filter(ours, sources)
        lines!(axs, 1:12, [band_mean(monthly[l][m, :, :]) for m in 1:12];
               color = get(NE_COLORS, l, :crimson), linewidth = 3, label = l)
    end
    vlines!(axs, @lift([Float64($imonth)]); color = (:black, 0.5), linewidth = 2)
    axislegend(axs; position = :rt, nbanks = 3, framevisible = false, labelsize = 11)

    out = joinpath(fig_dir, "fluxcomp_$(var)_monthly.mp4")
    record(fig, out, 1:12; framerate = 2) do m
        imonth[] = m
    end
    @info "saved $out"
end

if lowercase(get(ENV, "VIDEOS", "true")) == "true"
    for var in VARORDER
        monthly_video(var)
    end
end

@info "All figures in $fig_dir"
