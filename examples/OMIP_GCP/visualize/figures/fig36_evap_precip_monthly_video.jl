# Figure 36: monthly surface freshwater-flux video — tropical (|lat| ≤ 20°)
# longitude-index × latitude maps of evaporation, precipitation, and net
# freshwater flux, animated over every monthly snapshot in the case window.
# The animated counterpart of fig31; columns = cases, rows = E | P | net.
# Built with the Observable/@lift/CairoMakie.record idiom.
function fig36(caches, labels, cases; lat_range = (-20, 20), framerate = 4,
               reference_date = DateTime(1958, 1, 1))
    # Per-case geometry + monthly surface FieldTimeSeries.
    data = map(labels) do lab
        c = caches[lab]
        grid = get_field(c, :grid)
        Nx, Ny, _ = size(grid)
        φ2d   = [φnode(ii, jj, 1, grid, Center(), Center(), Center()) for ii in 1:Nx, jj in 1:Ny]
        φ1d   = vec(sum(φ2d; dims = 1) ./ Nx)
        jkeep = findall(φ -> lat_range[1] <= φ <= lat_range[2], φ1d)
        E = get_field(c, :evap_monthly_fts)
        P = get_field(c, :precip_monthly_fts)
        N = get_field(c, :wfo_monthly_fts)
        idx = in_window(E; start_time = c.start_time, stop_time = c.stop_time)
        (; lab, Nx, jkeep, φsub = φ1d[jkeep], E, P, N, idx, land = get_field(c, :land))
    end

    Nf = minimum(length(d.idx) for d in data)
    Nf == 0 && (@warn "fig36: no monthly surface frames"; return nothing)

    # Land-masked tropical slab of frame n for a given FTS.
    function slab(fts, d, n)
        A = Array(interior(fts[d.idx[n]])[:, :, 1])
        mask_land!(A, d.land)
        return A[:, d.jkeep]
    end

    # Fixed colorranges across frames: E and P share a positive range; the net
    # freshwater flux is diverging (balance).
    eprange = finite_extrema((slab(d.E, d, n) for d in data, n in 1:Nf)...,
                             (slab(d.P, d, n) for d in data, n in 1:Nf)...;
                             default = (0.0, 1e-4))
    nrange  = symmetric_extrema((slab(d.N, d, n) for d in data, n in 1:Nf)...; default = 1e-5)

    m   = Observable(1)
    fig = Figure(size = (700 * length(labels), 900), fontsize = 14)

    for (i, d) in enumerate(data)
        Eₙ = @lift slab(d.E, d, $m)
        Pₙ = @lift slab(d.P, d, $m)
        Nₙ = @lift slab(d.N, d, $m)

        axE = Axis(fig[1, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$(d.lab): Evaporation")
        hmE = heatmap!(axE, 1:d.Nx, d.φsub, Eₙ; colormap = :amp, colorrange = eprange, nan_color = :lightgray)
        Colorbar(fig[1, 2i], hmE; label = "kg/m²/s")

        axP = Axis(fig[2, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$(d.lab): Precipitation")
        hmP = heatmap!(axP, 1:d.Nx, d.φsub, Pₙ; colormap = :amp, colorrange = eprange, nan_color = :lightgray)
        Colorbar(fig[2, 2i], hmP; label = "kg/m²/s")

        axN = Axis(fig[3, 2i-1]; xlabel = "Longitude index", ylabel = "Latitude",
                   title = "$(d.lab): Net freshwater flux")
        hmN = heatmap!(axN, 1:d.Nx, d.φsub, Nₙ; colormap = :balance, colorrange = nrange, nan_color = :lightgray)
        Colorbar(fig[3, 2i], hmN; label = "kg/m²/s")
    end

    ref = data[1]
    title = @lift "Monthly surface freshwater fluxes — " *
                  "$(Dates.format(reference_date + Second(round(Int, ref.E.times[ref.idx[$m]])), "yyyy-mm"))"
    Label(fig[0, :], title; fontsize = 18)

    return savevideo(fig, "fig36_evap_precip_monthly.mp4", 1:Nf, mm -> (m[] = mm); framerate)
end
