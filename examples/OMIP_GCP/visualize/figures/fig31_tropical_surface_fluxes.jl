# Figure 31: tropical (|lat| <= 20°) maps of net surface fluxes into the
# ocean — net heat flux (W/m², row 1) and net freshwater flux (kg/m²/s,
# row 2), time-averaged over the case window. Rendered as longitude-index
# × latitude heatmaps on the native model grid (à la the calibration
# tropical-bias maps), with land NaN-masked to light grey.
function fig31(caches, labels, cases; lat_range = (-20, 20))
    fig = Figure(size = (700 * length(labels), 600), fontsize = 14)
    for (i, lab) in enumerate(labels)
        c    = caches[lab]
        grid = get_field(c, :grid)
        Nx, Ny, _ = size(grid)
        # Per-j representative latitude (φ is 1-D on lat-lon, 2-D on ORCA;
        # collapse to a 1-D y-axis the same way the calibration maps do).
        φ2d = [φnode(ii, jj, 1, grid, Center(), Center(), Center()) for ii in 1:Nx, jj in 1:Ny]
        φ1d = vec(sum(φ2d; dims = 1) ./ Nx)
        jkeep = findall(φᵢ -> lat_range[1] <= φᵢ <= lat_range[2], φ1d)
        φsub  = φ1d[jkeep]

        Q = get_field(c, :heat_flux)        # (Nx, Ny) W/m²,     land = NaN
        F = get_field(c, :freshwater_flux)  # (Nx, Ny) kg/m²/s,  land = NaN

        axQ = Axis(fig[1, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$lab: Net heat flux into ocean")
        hmQ = heatmap!(axQ, 1:Nx, φsub, Q[:, jkeep];
                       colormap = :balance, colorrange = (-200, 200),
                       nan_color = :lightgray)
        Colorbar(fig[1, 2i], hmQ; label = "W/m^2")

        axF = Axis(fig[2, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$lab: Net freshwater flux into ocean")
        hmF = heatmap!(axF, 1:Nx, φsub, F[:, jkeep];
                       colormap = :balance, colorrange = (-1e-5, 1e-5),
                       nan_color = :lightgray)
        Colorbar(fig[2, 2i], hmF; label = "kg/m^2/s")
    end
    savefig(fig, "fig31_tropical_surface_fluxes.png")
end
