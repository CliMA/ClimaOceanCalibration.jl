# Figure 32: tropical (|lat| <= 20°) horizontal-mean T and S profiles,
# time-averaged over the case window, with the WOA climatology profile
# (also tropical horizontal-mean, on the model grid) overlaid for
# reference. The fig17 counterpart restricted to the tropics.
function fig32(caches, labels, cases)
    fig = Figure(size = (900, 700), fontsize = 14)
    ax_temperature = Axis(fig[1, 1]; xlabel = "Temperature (deg C)", ylabel = "Depth (m)",
                          title = "Tropical (|lat|≤20°) horizontal-mean temperature")
    ax_salinity = Axis(fig[1, 2]; xlabel = "Salinity (PSU)", ylabel = "Depth (m)",
                       title = "Tropical (|lat|≤20°) horizontal-mean salinity")
    for (i, lab) in enumerate(labels)
        c = caches[lab]
        z = get_field(c, :depth)
        lines!(ax_temperature,
               get_field(c, :tropical_horizontal_mean_temperature_profile), z;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
        lines!(ax_salinity,
               get_field(c, :tropical_horizontal_mean_salinity_profile), z;
               color = case_colors[i], linewidth = CASE_LINEWIDTH, label = lab)
    end
    # WOA reference (tropical horizontal-mean) from the first case's grid;
    # identical across cases sharing a grid, so plot once.
    c0 = caches[first(labels)]
    z0 = get_field(c0, :depth)
    lines!(ax_temperature, get_field(c0, :tropical_woa_temperature_profile), z0;
           color = OBS_COLOR, linewidth = OBS_LINEWIDTH, linestyle = OBS_LINESTYLE, label = "WOA")
    lines!(ax_salinity, get_field(c0, :tropical_woa_salinity_profile), z0;
           color = OBS_COLOR, linewidth = OBS_LINEWIDTH, linestyle = OBS_LINESTYLE, label = "WOA")
    ylims!(ax_temperature, (-5500, 0))
    ylims!(ax_salinity, (-5500, 0))
    Legend(fig[2, :], ax_temperature; orientation = :horizontal, nbanks = length(labels) + 1)
    savefig(fig, "fig32_tropical_profiles.png")
end
