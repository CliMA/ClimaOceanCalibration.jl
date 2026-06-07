# Figure 20: T (row 1) and S (row 2) horizontal-mean drift as time × depth
# contours. Each per-case cell is a vertically-split panel: top half spans
# 0 → 1000 m (upper-ocean detail), bottom half spans 1000 m → bottom. The
# two sub-axes share the time axis and sit flush with no gap, so visually
# they read as a single panel with a piecewise-linear depth coordinate.
function fig21(caches, labels, cases)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 1200), fontsize = 14)

    upper = (-1000, 0)
    deep  = (-5500, -1000)

    # Build a vertically-split (top: 0-1000 m, bot: 1000 m-bottom) contour
    # panel at `fig[row, col]`. Returns the heatmap handle for the colorbar.
    function split_panel!(fig, row, col, t, z, data, levels, title_str)
        gl = fig[row, col] = GridLayout()
        ax_top = Axis(gl[1, 1]; title = title_str, ylabel = "Depth (m)")
        ax_bot = Axis(gl[2, 1]; xlabel = "Time (years)", ylabel = "Depth (m)")
        hm = contourf!(ax_top, t, z, data; levels = levels, colormap = :balance,
                        extendlow = :auto, extendhigh = :auto)
        contourf!(ax_bot, t, z, data; levels = levels, colormap = :balance,
                   extendlow = :auto, extendhigh = :auto)
        ylims!(ax_top, upper)
        ylims!(ax_bot, deep)
        linkxaxes!(ax_top, ax_bot)
        hidexdecorations!(ax_top; grid = false, ticks = false, minorticks = false)
        rowgap!(gl, 0)
        return hm
    end

    for (i, lab) in enumerate(labels)
        c  = caches[lab]
        z  = get_field(c, :depth)
        ΔT = get_field(c, :temperature_drift)
        ΔS = get_field(c, :salinity_drift)
        # Derive per-variable time axes so a panel never desynchronizes
        # from its data when `to_h` and `so_h` were written for different
        # iteration sets in the JLD2 averages file.
        tT = get_field(c, :to_h_fts).times ./ (365.25 * 24 * 3600)
        tS = get_field(c, :so_h_fts).times ./ (365.25 * 24 * 3600)

        hm_T = split_panel!(fig, 1, 2i-1, tT, z, ΔT,
                             temperature_drift_levels, "$lab: ΔT = T(t) − T_WOA (deg C)")
        Colorbar(fig[1, 2i], hm_T; label = "deg C")

        hm_S = split_panel!(fig, 2, 2i-1, tS, z, ΔS,
                             salinity_drift_levels, "$lab: ΔS = S(t) − S_WOA (PSU)")
        Colorbar(fig[2, 2i], hm_S; label = "PSU")
    end

    savefig(fig, "fig21_TS_drift_heatmap.png")
end

# Figure 27: upper-ocean (z ≥ z_min, default -500 m) counterpart to fig21.
# Same T (row 1) / S (row 2) horizontal-mean drift, but a single contour
# panel per case spanning the surface down to `z_min` only — no deep
# sub-axis. Intended as a focused upper-ocean view for calibration where
# only the near-surface drift matters.
function fig27(caches, labels, cases; z_min = -500)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 800), fontsize = 14)

    function upper_panel!(fig, row, col, t, z, data, levels, title_str)
        ax = Axis(fig[row, col]; title = title_str,
                   xlabel = "Time (years)", ylabel = "Depth (m)")
        hm = contourf!(ax, t, z, data; levels = levels, colormap = :balance,
                        extendlow = :auto, extendhigh = :auto)
        ylims!(ax, (z_min, 0))
        return hm
    end

    for (i, lab) in enumerate(labels)
        c  = caches[lab]
        z  = get_field(c, :depth)
        ΔT = get_field(c, :temperature_drift)
        ΔS = get_field(c, :salinity_drift)
        tT = get_field(c, :to_h_fts).times ./ (365.25 * 24 * 3600)
        tS = get_field(c, :so_h_fts).times ./ (365.25 * 24 * 3600)

        hm_T = upper_panel!(fig, 1, 2i-1, tT, z, ΔT,
                            temperature_drift_levels,
                            "$lab: ΔT = T(t) − T_WOA (deg C), upper $(abs(z_min)) m")
        Colorbar(fig[1, 2i], hm_T; label = "deg C")

        hm_S = upper_panel!(fig, 2, 2i-1, tS, z, ΔS,
                            salinity_drift_levels,
                            "$lab: ΔS = S(t) − S_WOA (PSU), upper $(abs(z_min)) m")
        Colorbar(fig[2, 2i], hm_S; label = "PSU")
    end

    savefig(fig, "fig27_TS_drift_heatmap_upper$(abs(z_min))m.png")
end
