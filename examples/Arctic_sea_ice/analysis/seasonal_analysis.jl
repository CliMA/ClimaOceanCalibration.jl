# Seasonal-cycle checks of a pan-Arctic run: ice extent and volume against GLORYS, surface means by region,
# drift of the monthly 3D T and S, and the daily extrema from the slurm log.
# Usage: julia --project=.. seasonal_analysis.jl <figure dir> <run dir> [<slurm log>]
# Needs glorys_ice.jld2 in the run dir (from glorys_ice_series.jl).

using Oceananigans, Oceananigans.Units, CairoMakie, Statistics, Printf, Dates
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_inactive_node
const JLD2 = Oceananigans.OutputWriters.JLD2

figdir = mkpath(ARGS[1])
rundir = ARGS[2]
logfile = length(ARGS) >= 3 ? ARGS[3] : nothing
day₀ = DateTime(1993, 1, 2)
const H = 7   # halo width of the saved fields

oceanfile = joinpath(rundir, "arctic_ocean_surface.jld2")
icefile   = joinpath(rundir, "arctic_sea_ice_surface.jld2")

iterations(f, name) = sort(parse.(Int, filter(k -> all(isdigit, k), collect(keys(f["timeseries/$name"])))))
function series(file, name)
    JLD2.jldopen(file) do f
        its = iterations(f, name)
        [f["timeseries/t/$it"] for it in its], [Array(f["timeseries/$name/$it"])[H+1:end-H, H+1:end-H, 1] for it in its]
    end
end
has(file, name) = JLD2.jldopen(f -> haskey(f["timeseries"], name), file)

grid = JLD2.jldopen(f -> f["serialized/grid"], icefile)
Nx, Ny, Nz = size(grid)
wet  = [!immersed_inactive_node(i, j, Nz, grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
area = [Azᶜᶜᶜ(i, j, Nz, grid) for i in 1:Nx, j in 1:Ny]

times, he = series(icefile, "he")
ℵ = has(icefile, "ℵ") ? series(icefile, "ℵ")[2] : nothing
T = series(oceanfile, "T")[2]
S = series(oceanfile, "S")[2]
U = series(oceanfile, "𝒱")[2]
tdays = (times .- times[1]) ./ 1days
dates = day₀ .+ Day.(round.(Int, tdays))
clean(x) = isnan(x) ? 0.0 : x

# Ice extent and volume
extent = isnothing(ℵ) ? fill(NaN, length(tdays)) : [sum(area[wet .& (clean.(a) .> 0.15)]) for a in ℵ]
volume = [sum((clean.(v) .* area)[wet]) for v in he]
g = JLD2.jldopen(joinpath(rundir, "glorys_ice.jld2"))
gdays = (g["times"] .- g["times"][1]) ./ 1days
gextent, gvolume = g["extent"], g["volume"]
ghe = g["volume_per_area"]
surface_file = joinpath(rundir, "glorys_surface_series.jld2")
glorys_surface = isfile(surface_file) ? JLD2.jldopen(f -> (times = f["times"], T = f["T"], S = f["S"]), surface_file) : nothing

# Surface means by region (interior = more than 5 cells from an open edge; rims = the 5 cells next to each)
rim = 5
interior_mask = copy(wet); interior_mask[1:rim, :] .= false; interior_mask[:, 1:rim] .= false; interior_mask[:, Ny-rim+1:Ny] .= false
regions = (interior = interior_mask, atlantic = wet .& [j <= rim for i in 1:Nx, j in 1:Ny],
           pacific = wet .& [j > Ny - rim for i in 1:Nx, j in 1:Ny], alaska = wet .& [i <= rim for i in 1:Nx, j in 1:Ny])
means = Dict(name => (T = [mean(x[m]) for x in T], S = [mean(x[m]) for x in S],
                      U = [mean(x[m]) for x in U], Umax = [maximum(x[m]) for x in U]) for (name, m) in pairs(regions))
if !isnothing(glorys_surface)
    sdays = (glorys_surface.times .- glorys_surface.times[1]) ./ 1days
    gmeans = Dict(name => (T = [mean(glorys_surface.T[:, :, k][m]) for k in eachindex(sdays)],
                           S = [mean(glorys_surface.S[:, :, k][m]) for k in eachindex(sdays)]) for (name, m) in pairs(regions))
end

# Daily extrema from the slurm log: max |u|, |v|, |w|, T and S extrema, max h
logrows = []
if !isnothing(logfile)
    for line in eachline(logfile)
        m = match(r"time: (\d+(?:\.\d+)?) (day|days|hour|hours), iter: \d+, extrema\(T, S\): \((-?\d+\.\d), (-?\d+\.\d)\) °C \((-?\d+\.\d), (-?\d+\.\d)\) psu, extrema\(u\) \(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) max\(h\): ([\d.]+) m", line)
        isnothing(m) && continue
        t = parse(Float64, m[1]) * (startswith(m[2], "day") ? 1 : 1/24)
        push!(logrows, (t, parse.(Float64, m.captures[3:10])...))
    end
    isempty(logrows) || (t₀ = logrows[1][1]; logrows = [(r[1] - t₀, r[2:end]...) for r in logrows])
end

fig = Figure(size = (1500, 1000))
ax = Axis(fig[1, 1], title = "Sea-ice extent [million km²]", xlabel = "days from 2 Jan 1993")
lines!(ax, tdays, extent ./ 1e12, label = "model"); lines!(ax, gdays, gextent ./ 1e12, linestyle = :dash, color = :black, label = "GLORYS"); axislegend(ax)
ax = Axis(fig[1, 2], title = "Sea-ice volume [thousand km³]", xlabel = "days")
lines!(ax, tdays, volume ./ 1e12); lines!(ax, gdays, gvolume ./ 1e12, linestyle = :dash, color = :black)
ax = Axis(fig[1, 3], title = "mean surface speed [m s⁻¹]", xlabel = "days")
for (name, m) in pairs(means); lines!(ax, tdays, m.U, label = string(name)); end; axislegend(ax, position = :lt)
ax = Axis(fig[2, 1], title = "mean SST [°C]", xlabel = "days")
for (name, m) in pairs(means); lines!(ax, tdays, m.T, label = string(name)); end
isnothing(glorys_surface) || for (name, m) in pairs(gmeans); lines!(ax, sdays, m.T, linestyle = :dash, color = :black); end
ax = Axis(fig[2, 2], title = "mean SSS [psu]", xlabel = "days")
for (name, m) in pairs(means); lines!(ax, tdays, m.S, label = string(name)); end
isnothing(glorys_surface) || for (name, m) in pairs(gmeans); lines!(ax, sdays, m.S, linestyle = :dash, color = :black); end
ax = Axis(fig[2, 3], title = "max surface speed [m s⁻¹]", xlabel = "days")
for (name, m) in pairs(means); lines!(ax, tdays, m.Umax, label = string(name)); end
if !isempty(logrows)
    t = first.(logrows)
    ax = Axis(fig[3, 1], title = "log: max |u|, |v|, |w| [m s⁻¹]", xlabel = "days", yscale = log10)
    lines!(ax, t, max.(getindex.(logrows, 6), 1e-4), label = "|u|"); lines!(ax, t, max.(getindex.(logrows, 7), 1e-4), label = "|v|"); lines!(ax, t, max.(getindex.(logrows, 8), 1e-4), label = "|w|"); axislegend(ax)
    ax = Axis(fig[3, 2], title = "log: T and S extrema", xlabel = "days")
    lines!(ax, t, getindex.(logrows, 2), label = "min T"); lines!(ax, t, getindex.(logrows, 3), label = "max T"); lines!(ax, t, getindex.(logrows, 4), label = "min S"); lines!(ax, t, getindex.(logrows, 5), label = "max S"); axislegend(ax, position = :lt)
    ax = Axis(fig[3, 3], title = "log: max ice thickness [m]", xlabel = "days")
    lines!(ax, t, getindex.(logrows, 9))
end
save(joinpath(figdir, "seasonal_timeseries.png"), fig)

nearest(ds, target) = findmin(abs.(ds .- target))[2]
picks = [d for d in (60, 150, 240, 330) if d <= maximum(tdays)]
isempty(picks) && (picks = [maximum(tdays)])

# Ice maps: model and GLORYS
fig = Figure(size = (400 * length(picks) + 100, 850))
for (c, d) in enumerate(picks)
    n = nearest(tdays, d); m = nearest(gdays, d)
    A = clean.(he[n]); A[.!wet] .= NaN
    B = copy(ghe[:, :, m]); B[.!wet] .= NaN
    ax = Axis(fig[1, c], title = "model, $(Dates.format(dates[n], "d u"))", aspect = DataAspect()); hm = heatmap!(ax, A, colormap = Reverse(:blues), colorrange = (0, 3), nan_color = :gray80)
    ax = Axis(fig[2, c], title = "GLORYS", aspect = DataAspect()); heatmap!(ax, B, colormap = Reverse(:blues), colorrange = (0, 3), nan_color = :gray80)
    c == length(picks) && Colorbar(fig[1:2, c+1], hm, label = "ice volume per area [m]")
end
save(joinpath(figdir, "seasonal_ice_maps.png"), fig)

# Surface maps at the same dates
fig = Figure(size = (400 * length(picks) + 100, 850))
for (c, d) in enumerate(picks)
    n = nearest(tdays, d)
    A = copy(T[n]); A[.!wet] .= NaN
    B = copy(S[n]); B[.!wet] .= NaN
    ax = Axis(fig[1, c], title = "SST, $(Dates.format(dates[n], "d u"))", aspect = DataAspect()); hm1 = heatmap!(ax, A, colormap = :thermal, colorrange = (-2, 12), nan_color = :gray80)
    ax = Axis(fig[2, c], title = "SSS", aspect = DataAspect()); hm2 = heatmap!(ax, B, colormap = :haline, colorrange = (28, 35.5), nan_color = :gray80)
    if c == length(picks); Colorbar(fig[1, c+1], hm1, label = "°C"); Colorbar(fig[2, c+1], hm2, label = "psu"); end
end
save(joinpath(figdir, "seasonal_surface_maps.png"), fig)

# Monthly 3D T and S: domain-mean profiles and their change from the first snapshot
monthly = joinpath(rundir, "arctic_ocean_monthly.jld2")
if isfile(monthly)
    z = znodes(grid, Center(), Center(), Center())
    wet3 = [!immersed_inactive_node(i, j, k, grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    JLD2.jldopen(monthly) do f
        its = iterations(f, "T")
        mtimes = [f["timeseries/t/$it"] for it in its]
        profile(name, it) = (A = Array(f["timeseries/$name/$it"])[H+1:end-H, H+1:end-H, H+1:end-H]; [mean(A[:, :, k][wet3[:, :, k]]) for k in 1:Nz])
        T₀, S₀ = profile("T", its[1]), profile("S", its[1])
        fig = Figure(size = (1200, 600))
        ax1 = Axis(fig[1, 1], title = "domain-mean T change from the start [°C]", ylabel = "z [m]")
        ax2 = Axis(fig[1, 2], title = "domain-mean S change from the start [psu]")
        for (n, it) in enumerate(its[2:end])
            lines!(ax1, profile("T", it) .- T₀, z, label = @sprintf("day %.0f", mtimes[n+1] / 1days))
            lines!(ax2, profile("S", it) .- S₀, z)
        end
        axislegend(ax1, position = :rb)
        save(joinpath(figdir, "seasonal_profile_drift.png"), fig)
    end
end

open(joinpath(figdir, "seasonal_summary.txt"), "w") do io
    for (k, d) in enumerate(tdays)
        m = nearest(gdays, d)
        @printf(io, "day %5.1f  extent model %.2f GLORYS %.2f Mkm²  volume model %.2f GLORYS %.2f thousand km³  interior SST %.2f SSS %.3f speed %.3f  atlantic SST %.2f SSS %.3f  pacific SST %.2f SSS %.3f  alaska SST %.2f SSS %.3f",
                d, extent[k] / 1e12, gextent[m] / 1e12, volume[k] / 1e12, gvolume[m] / 1e12,
                means[:interior].T[k], means[:interior].S[k], means[:interior].U[k],
                means[:atlantic].T[k], means[:atlantic].S[k], means[:pacific].T[k], means[:pacific].S[k], means[:alaska].T[k], means[:alaska].S[k])
        if !isnothing(glorys_surface)
            n = nearest(sdays, d)
            @printf(io, "  minus GLORYS: interior %+.2f %+.3f  atlantic %+.2f %+.3f  pacific %+.2f %+.3f  alaska %+.2f %+.3f",
                    means[:interior].T[k] - gmeans[:interior].T[n], means[:interior].S[k] - gmeans[:interior].S[n],
                    means[:atlantic].T[k] - gmeans[:atlantic].T[n], means[:atlantic].S[k] - gmeans[:atlantic].S[n],
                    means[:pacific].T[k] - gmeans[:pacific].T[n], means[:pacific].S[k] - gmeans[:pacific].S[n],
                    means[:alaska].T[k] - gmeans[:alaska].T[n], means[:alaska].S[k] - gmeans[:alaska].S[n])
        end
        println(io)
    end
end
@info "Figures written to $figdir"
