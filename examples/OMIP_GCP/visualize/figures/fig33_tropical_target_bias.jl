# Figure 33: the calibration target itself (cf. plot_tropical_target_bias in
# examples/CATKE_GM_calibration/calibration_plots.jl). Tropical (|lat| <= 20°)
# lon-index × latitude heatmap of the T and S bias vs WOA, vertically averaged
# over the upper |z_min| m (default 400 m), time-averaged over the case window.
# Land cells are NaN-masked to light grey.
function fig33(caches, labels, cases; lat_range = (-20, 20), z_min = -400)
    fig = Figure(size = (700 * length(labels), 600), fontsize = 14)
    for (i, lab) in enumerate(labels)
        c    = caches[lab]
        grid = get_field(c, :grid)
        Nx, Ny, _ = size(grid)
        z      = get_field(c, :depth)
        kkeep  = findall(zᵢ -> zᵢ >= z_min, z)
        mask3d = get_field(c, :ocean_mask_3d)

        Tm3 = get_field(c, :time_mean_temperature_3d)
        Sm3 = get_field(c, :time_mean_salinity_3d)
        Tw3 = get_field(c, :woa_temperature)
        Sw3 = get_field(c, :woa_salinity)

        # NaN-safe vertical mean over the upper |z_min| m, ocean-masked so
        # land columns drop out (matches the calibration `vmean_top`).
        function vmean_top(d)
            out = fill(NaN, Nx, Ny)
            for jj in 1:Ny, ii in 1:Nx
                num = 0.0; den = 0
                for k in kkeep
                    mask3d[ii, jj, k] > 0 || continue
                    v = d[ii, jj, k]
                    isfinite(v) || continue
                    num += v; den += 1
                end
                den > 0 && (out[ii, jj] = num / den)
            end
            return out
        end

        Tbias = vmean_top(Tm3) .- vmean_top(Tw3)
        Sbias = vmean_top(Sm3) .- vmean_top(Sw3)

        # Per-j representative latitude (φ is 1-D on lat-lon, 2-D on ORCA).
        φ2d   = [φnode(ii, jj, 1, grid, Center(), Center(), Center()) for ii in 1:Nx, jj in 1:Ny]
        φ1d   = vec(sum(φ2d; dims = 1) ./ Nx)
        jkeep = findall(φᵢ -> lat_range[1] <= φᵢ <= lat_range[2], φ1d)
        φsub  = φ1d[jkeep]

        axT = Axis(fig[1, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$lab: T bias (model − WOA), upper $(abs(z_min)) m")
        hmT = heatmap!(axT, 1:Nx, φsub, Tbias[:, jkeep];
                       colormap = :balance, colorrange = (-3.5, 3.5),
                       nan_color = :lightgray)
        Colorbar(fig[1, 2i], hmT; label = "°C")

        axS = Axis(fig[2, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$lab: S bias (model − WOA), upper $(abs(z_min)) m")
        hmS = heatmap!(axS, 1:Nx, φsub, Sbias[:, jkeep];
                       colormap = :balance, colorrange = (-1.5, 1.5),
                       nan_color = :lightgray)
        Colorbar(fig[2, 2i], hmS; label = "PSU")
    end
    savefig(fig, "fig33_tropical_target_bias.png")
end
