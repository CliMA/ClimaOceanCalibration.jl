# Figure 28: tropical-band (|lat| <= 20°) counterpart of fig21. T (row 1)
# and S (row 2) horizontal-mean drift over the tropics as time × depth
# contours, with the same vertically-split panel (top: 0 → 1000 m, bottom:
# 1000 m → bottom) so the upper ocean is resolved. The horizontal mean is
# taken over tropical cells only (see `:tropical_temperature_drift`), unlike
# fig21 which uses the global `to_h`/`so_h` profiles.
function fig28(caches, labels, cases)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.1, 0.1; length = 21)
    fig = Figure(size = (900 * ncases, 1200), fontsize = 14)

    upper = (-1000, 0)
    deep  = (-5500, -1000)

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
        ΔT = get_field(c, :tropical_temperature_drift)
        ΔS = get_field(c, :tropical_salinity_drift)
        # Tropical drift is computed from the 3-D `to_fts`/`so_fts` series,
        # so its time axis comes from those (not the global `to_h`/`so_h`).
        tT = get_field(c, :to_fts).times ./ (365.25 * 24 * 3600)
        tS = get_field(c, :so_fts).times ./ (365.25 * 24 * 3600)

        hm_T = split_panel!(fig, 1, 2i-1, tT, z, ΔT,
                             temperature_drift_levels, "$lab: ΔT (deg C), tropics |lat|≤20°")
        Colorbar(fig[1, 2i], hm_T; label = "deg C")

        hm_S = split_panel!(fig, 2, 2i-1, tS, z, ΔS,
                             salinity_drift_levels, "$lab: ΔS (PSU), tropics |lat|≤20°")
        Colorbar(fig[2, 2i], hm_S; label = "PSU")
    end

    savefig(fig, "fig28_tropical_TS_drift_heatmap.png")
end

# Figure 29: upper-ocean (z ≥ z_min, default -500 m) tropical drift —
# fig28 restricted to the surface down to `z_min`, single contour panel per
# case (no deep sub-axis). The tropical, upper-ocean focus matching fig27.
function fig29(caches, labels, cases; z_min = -500)
    ncases = length(labels)
    temperature_drift_levels = range(-1.6, 1.6; length = 17)
    salinity_drift_levels    = range(-0.2, 0.2; length = 21)
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
        ΔT = get_field(c, :tropical_temperature_drift)
        ΔS = get_field(c, :tropical_salinity_drift)
        tT = get_field(c, :to_fts).times ./ (365.25 * 24 * 3600)
        tS = get_field(c, :so_fts).times ./ (365.25 * 24 * 3600)

        hm_T = upper_panel!(fig, 1, 2i-1, tT, z, ΔT,
                            temperature_drift_levels,
                            "$lab: ΔT (deg C), tropics |lat|≤20°, upper $(abs(z_min)) m")
        Colorbar(fig[1, 2i], hm_T; label = "deg C")

        hm_S = upper_panel!(fig, 2, 2i-1, tS, z, ΔS,
                            salinity_drift_levels,
                            "$lab: ΔS (PSU), tropics |lat|≤20°, upper $(abs(z_min)) m")
        Colorbar(fig[2, 2i], hm_S; label = "PSU")
    end

    savefig(fig, "fig29_tropical_TS_drift_heatmap_upper$(abs(z_min))m.png")
end
