# Figure 35: seasonal-cycle zonal-mean video — GLOBAL (lat −90..90), full depth.
# One .mp4 per case: columns WOA | Simulation | (Model − WOA), rows T | S |
# buoyancy, animated over every monthly snapshot in the case window. The global
# counterpart of fig34. See `render_zonal_monthly_video` in common.jl.
function fig35(caches, labels, cases)
    for lab in labels
        tag = replace(lab, r"[^A-Za-z0-9]+" => "_")
        render_zonal_monthly_video(caches[lab], lab;
                                   lat_range = (-90, 90),
                                   z_min     = -5500,
                                   filename  = "fig35_zonal_monthly_global_$(tag).mp4")
    end
end
