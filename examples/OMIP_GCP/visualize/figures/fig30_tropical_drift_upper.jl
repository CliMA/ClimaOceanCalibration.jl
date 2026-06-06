# Figure 30: tropical (|lat| <= 20°) upper-ocean (z >= z_min, default
# -400 m) horizontal-mean T and S drift vs time, referenced to the first
# snapshot. The fig16 line-plot counterpart of the tropical drift
# heatmaps (fig28/29): the (Nt, Nz) tropical drift profile is vertically
# averaged over the upper |z_min| m to a single time series per case.
function fig30(caches, labels, cases; z_min = -400)
    fig = Figure(size = (600 + 200 * length(labels), 450), fontsize = 14)
    ax_temperature = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "ΔT (deg C)",
                          title = "Tropical (|lat|≤20°) upper $(abs(z_min)) m temperature drift")
    ax_salinity = Axis(fig[1, 2]; xlabel = "Time (years)", ylabel = "ΔS (PSU)",
                       title = "Tropical (|lat|≤20°) upper $(abs(z_min)) m salinity drift")
    for (i, lab) in enumerate(labels)
        c  = caches[lab]
        z  = get_field(c, :depth)
        kkeep = findall(zᵢ -> zᵢ >= z_min, z)
        ΔT = get_field(c, :tropical_temperature_drift)   # (Nt, Nz)
        ΔS = get_field(c, :tropical_salinity_drift)
        # Tropical drift is computed from the 3-D to_fts/so_fts series,
        # so its time axis comes from those (cf. fig28).
        tT = get_field(c, :to_fts).times ./ (365.25 * 24 * 3600)
        tS = get_field(c, :so_fts).times ./ (365.25 * 24 * 3600)
        # Unweighted mean over upper-ocean levels (NaN-safe), matching the
        # unweighted horizontal mean used to build the drift profile.
        ΔT_upper = [nanmean(@view ΔT[n, kkeep]) for n in 1:length(tT)]
        ΔS_upper = [nanmean(@view ΔS[n, kkeep]) for n in 1:length(tS)]
        lines!(ax_temperature, tT, ΔT_upper;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
        lines!(ax_salinity, tS, ΔS_upper;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end
    Legend(fig[1, 3], ax_temperature)
    savefig(fig, "fig30_tropical_drift_upper$(abs(z_min))m.png")
end
