# seasonal_plots.jl
# Per-member seasonal-cycle diagnostic videos for the seasonal CATKE+GM
# calibration:
#   * plot_member_seasonal_video    — zonal-mean T, S, buoyancy (3 rows),
#     columns WOA | Simulation | (Model − WOA), 12 monthly frames.
#   * plot_member_seasonal_EP_video — model-only tropical maps of evaporation,
#     precipitation, and net freshwater flux (3 rows), 12 monthly frames.
# Both use the Observable/@lift/CairoMakie.record idiom. Reads the member's
# monthly T,S,b and E/P through data_processing_seasonal.jl and the OMIP-side
# field layout; the WOA T/S/b reference comes from the precomputed 2-D zonal
# arrays in the seasonal target file.

using CairoMakie
using JLD2
using Oceananigans
using Oceananigans.Grids: φnode

isdefined(@__MODULE__, :seasonal_zonal_TS_2d) ||
    include(joinpath(@__DIR__, "data_processing_seasonal.jl"))

# Finite (min,max) / symmetric ±max(|·|) over arrays, with fallbacks. Local
# copies (the visualize/common.jl versions aren't on the calibration path).
function _finite_extrema(arrays...; default = (-1.0, 1.0))
    lo = Inf; hi = -Inf
    for A in arrays, x in A
        isfinite(x) && (lo = min(lo, x); hi = max(hi, x))
    end
    return lo <= hi ? (lo, hi) : default
end
function _symmetric_extrema(arrays...; default = 1.0)
    a = 0.0
    for A in arrays, x in A
        isfinite(x) && (a = max(a, abs(x)))
    end
    a = a > 0 ? a : default
    return (-a, a)
end

"""
    plot_member_seasonal_video(member_dir, filename_prefix, woa_file,
                               lat_range, z_min, out_path;
                               framerate, plot_z_min)

Render the member's full-run zonal-mean T/S/buoyancy seasonal cycle vs the WOA
monthly climatology into `out_path` (.mp4). One frame per monthly snapshot in the
member file (all years of the run). Simulations start in January with monthly
output, so frame `f` is calendar month `mod1(f, 12)`; its WOA reference is that
climatological month (WOA Monthly is a 12-month climatology, indexed cyclically,
so any number of model years is covered).

`z_min` is the calibration cutoff (kept for the call signature); the video depth
extent is set by `plot_z_min` (default −400 m). Returns the output path.
"""
function plot_member_seasonal_video(member_dir::AbstractString,
                                    filename_prefix::AbstractString,
                                    woa_file::AbstractString,
                                    lat_range, z_min, out_path::AbstractString;
                                    framerate = 2, plot_z_min = -400.0)
    # Member zonal-mean T,S,b for ALL monthly snapshots, same regrid pipeline as
    # the obs map.
    zTm, zSm, zBm, latitude, depth, times = member_zonal_TSB_2d(member_dir, filename_prefix;
                                                                all_months = true)

    # WOA reference 2-D zonal arrays + its own (shallow) depth axis. The WOA grid
    # is the top cells of the model grid, so its z ≥ plot_z_min cells coincide with
    # the member's; slicing each by its own depth gives identical cells/shapes.
    woa = jldopen(woa_file, "r") do f
        (zT = f["zonal_T"], zS = f["zonal_S"], zb = f["zonal_b"], depth = f["depth"])
    end
    n_woa_months = length(woa.zT)

    j  = findall(φ -> lat_range[1] <= φ <= lat_range[2], latitude)
    k  = findall(z -> z >= plot_z_min, depth)          # member depth levels
    kw = findall(z -> z >= plot_z_min, woa.depth)      # WOA (shallow) depth levels — same cells
    length(k) == length(kw) || error("plot_member_seasonal_video: member ($(length(k))) and \
        WOA ($(length(kw))) have different #levels above $(plot_z_min) m; grids don't coincide there")
    lat_sub = latitude[j]
    z_sub   = depth[k]
    sub(A)  = Array(A[j, k])                       # member arrays
    subw(A) = Array(A[j, kw])                      # WOA arrays

    N = length(zTm)
    # Simulations start in January with monthly output, so frame f is calendar
    # month mod1(f, 12) of model year div(f-1, 12)+1. WOA Monthly is a 12-month
    # climatology indexed by calendar month — this is the cyclical WOA index.
    wmonth = [mod1(f, 12) for f in 1:N]
    yr     = [div(f - 1, 12) + 1 for f in 1:N]

    Trange  = (10, 30); Srange = (34.5, 36)
    Brange  = _finite_extrema((subw(woa.zb[mc]) for mc in 1:n_woa_months)...; default = (-0.04, 0.02))
    Tdrange = (-5.0, 5.0); Sdrange = (-0.5, 0.5)
    Bdrange = (-0.0075, 0.0075)

    m  = Observable(1)
    wm = @lift wmonth[$m]   # WOA climatological month for the current frame
    Tw = @lift subw(woa.zT[$wm]); Tm = @lift sub(zTm[$m]); Td = @lift sub(zTm[$m]) .- subw(woa.zT[$wm])
    Sw = @lift subw(woa.zS[$wm]); Sm = @lift sub(zSm[$m]); Sd = @lift sub(zSm[$m]) .- subw(woa.zS[$wm])
    Bw = @lift subw(woa.zb[$wm]); Bm = @lift sub(zBm[$m]); Bd = @lift sub(zBm[$m]) .- subw(woa.zb[$wm])

    fig = Figure(size = (1500, 1150), fontsize = 15)
    rows = ((Tw, Tm, Td, Trange, Tdrange, :turbo, "T (°C)"),
            (Sw, Sm, Sd, Srange, Sdrange, :turbo, "S (psu)"),
            (Bw, Bm, Bd, Brange, Bdrange, :turbo, "b (m/s²)"))
    coltitle(r, txt) = r == 1 ? txt : ""
    for (r, (w, s, d, rng, drng, cmap, unit)) in enumerate(rows)
        axw = Axis(fig[r, 1]; xlabel = "Latitude", ylabel = "Depth (m)", title = coltitle(r, "WOA"))
        axs = Axis(fig[r, 2]; xlabel = "Latitude", title = coltitle(r, "Simulation"))
        hm  = heatmap!(axw, lat_sub, z_sub, w; colormap = cmap, colorrange = rng, nan_color = :lightgray)
              heatmap!(axs, lat_sub, z_sub, s; colormap = cmap, colorrange = rng, nan_color = :lightgray)
        Colorbar(fig[r, 3], hm; label = unit)
        axd = Axis(fig[r, 4]; xlabel = "Latitude", title = coltitle(r, "Model − WOA"))
        hmd = heatmap!(axd, lat_sub, z_sub, d; colormap = :balance, colorrange = drng, nan_color = :lightgray)
        Colorbar(fig[r, 5], hmd; label = "Δ$unit")
        for ax in (axw, axs, axd)
            ylims!(ax, (plot_z_min, 0))
        end
    end

    title = @lift "Zonal-mean seasonal cycle — model year $(yr[$m]), month $(wmonth[$m])"
    Label(fig[0, :], title; fontsize = 20)

    CairoMakie.record(fig, out_path, 1:N; framerate) do mm
        m[] = mm
    end
    return out_path
end

"""
    plot_member_seasonal_EP_video(member_dir, filename_prefix, lat_range, out_path;
                                  framerate)

Render a model-only animated tropical map of monthly evaporation, precipitation,
and net freshwater flux (3 rows), over ALL of the member's monthly snapshots
(every year of the run). No observational reference (there is no WOA E/P target).
"""
function plot_member_seasonal_EP_video(member_dir::AbstractString,
                                       filename_prefix::AbstractString,
                                       lat_range, out_path::AbstractString;
                                       framerate = 2)
    path = joinpath(member_dir, "$(filename_prefix)_monthly_EP.jld2")
    isfile(path) || error("plot_member_seasonal_EP_video: missing $path")
    Efts = FieldTimeSeries(path, "evap")
    Pfts = FieldTimeSeries(path, "precip")
    Wfts = FieldTimeSeries(path, "wfo")
    nt = length(Efts.times)
    nt >= 1 || error("plot_member_seasonal_EP_video: no monthly E/P snapshots")
    window = 1:nt

    grid = Efts.grid
    Nx, Ny, _ = size(grid)
    φ2d   = [φnode(ii, jj, 1, grid, Center(), Center(), Center()) for ii in 1:Nx, jj in 1:Ny]
    φ1d   = vec(sum(φ2d; dims = 1) ./ Nx)
    jkeep = findall(φ -> lat_range[1] <= φ <= lat_range[2], φ1d)
    φsub  = φ1d[jkeep]
    land  = grid isa ImmersedBoundaryGrid ?
            (Array(interior(grid.immersed_boundary.bottom_height, :, :, 1)) .>= 0) :
            falses(Nx, Ny)

    function slab(fts, n)
        A = Array(interior(fts[window[n]])[:, :, 1])
        A[land] .= NaN
        return A[:, jkeep]
    end

    N = nt
    # Simulations start in January with monthly output: frame f is calendar month
    # mod1(f, 12) of model year div(f-1, 12)+1.
    emonth = [mod1(f, 12) for f in 1:N]
    yr     = [div(f - 1, 12) + 1 for f in 1:N]

    # eprange = _finite_extrema((slab(Efts, n) for n in 1:N)..., (slab(Pfts, n) for n in 1:N)...;
    #                           default = (0.0, 1e-4))
    # nrange  = _symmetric_extrema((slab(Wfts, n) for n in 1:N)...; default = 1e-5)

    eprange = (0, 3e-4)  # fixed range for E/P (kg/m²/s)
    nrange  = (-1.5e-5, 1.5e-5)  # fixed range for net freshwater flux (kg/m²/s)

    m  = Observable(1)
    En = @lift slab(Efts, $m)
    Pn = @lift slab(Pfts, $m)
    Wn = @lift slab(Wfts, $m)

    fig = Figure(size = (900, 950), fontsize = 14)
    axE = Axis(fig[1, 1]; xlabel = "Longitude index", ylabel = "Latitude", title = "Evaporation")
    hmE = heatmap!(axE, 1:Nx, φsub, En; colormap = :amp, colorrange = eprange, nan_color = :lightgray)
    Colorbar(fig[1, 2], hmE; label = "kg/m²/s")
    axP = Axis(fig[2, 1]; xlabel = "Longitude index", ylabel = "Latitude", title = "Precipitation")
    hmP = heatmap!(axP, 1:Nx, φsub, Pn; colormap = :amp, colorrange = eprange, nan_color = :lightgray)
    Colorbar(fig[2, 2], hmP; label = "kg/m²/s")
    axW = Axis(fig[3, 1]; xlabel = "Longitude index", ylabel = "Latitude", title = "Net freshwater flux")
    hmW = heatmap!(axW, 1:Nx, φsub, Wn; colormap = :balance, colorrange = nrange, nan_color = :lightgray)
    Colorbar(fig[3, 2], hmW; label = "kg/m²/s")

    title = @lift "Monthly surface freshwater fluxes — model year $(yr[$m]), month $(emonth[$m])"
    Label(fig[0, :], title; fontsize = 18)

    CairoMakie.record(fig, out_path, 1:N; framerate) do mm
        m[] = mm
    end
    return out_path
end
