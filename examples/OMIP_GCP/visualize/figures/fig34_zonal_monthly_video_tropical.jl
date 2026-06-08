# Figure 34: seasonal-cycle zonal-mean video — TROPICAL band (|lat| ≤ 20°),
# upper 500 m. One .mp4 per case: columns WOA | Simulation | (Model − WOA),
# rows T | S | buoyancy, animated over every monthly snapshot in the case
# window. The WOA reference cycles through its 12-month climatology by calendar
# month. See `render_zonal_monthly_video` in common.jl for the implementation.
function fig34(caches, labels, cases)
    for lab in labels
        tag = replace(lab, r"[^A-Za-z0-9]+" => "_")
        render_zonal_monthly_video(caches[lab], lab;
                                   lat_range = (-20, 20),
                                   z_min     = -500,
                                   filename  = "fig34_zonal_monthly_tropical_$(tag).mp4")
    end
end
