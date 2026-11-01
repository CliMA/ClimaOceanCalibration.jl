# Daily videos of a pan-Arctic run against GLORYS: ice volume per area, SST and SSS side by side, and the surface speed.
# Usage: julia --project=.. seasonal_movies.jl <video dir> <run dir>
# Needs glorys_ice.jld2 and glorys_surface_series.jld2 in the run dir (glorys_ice_series.jl, glorys_surface_series.jl).

using Oceananigans, Oceananigans.Units, CairoMakie, Dates
using Oceananigans.ImmersedBoundaries: immersed_inactive_node
include("saved_series.jl")

videodir = mkpath(ARGS[1])
rundir = ARGS[2]
day₀ = DateTime(1993, 1, 2)

oceanfile = joinpath(rundir, "arctic_ocean_surface.jld2")
icefile   = joinpath(rundir, "arctic_sea_ice_surface.jld2")

grid = JLD2.jldopen(f -> f["serialized/grid"], icefile)
Nx, Ny, Nz = size(grid)
wet = [!immersed_inactive_node(i, j, Nz, grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
masked(A) = (B = copy(A); B[.!wet] .= NaN; B)
clean(x) = isnan(x) ? 0f0 : x

times, he = series(icefile, "he"; float = Float32)
T = series(oceanfile, "T"; float = Float32)[2]
S = series(oceanfile, "S"; float = Float32)[2]
U = series(oceanfile, "𝒱"; float = Float32)[2]
tdays = round.(Int, (times .- times[1]) ./ 1days)
dates = day₀ .+ Day.(tdays)

g = JLD2.jldopen(joinpath(rundir, "glorys_ice.jld2"))
gdays = round.(Int, (g["times"] .- g["times"][1]) ./ 1days)
ghe = g["volume_per_area"]
gs = JLD2.jldopen(joinpath(rundir, "glorys_surface_series.jld2"))
sdays = round.(Int, (gs["times"] .- gs["times"][1]) ./ 1days)
gT, gS = gs["T"], gs["S"]
nearest(ds, d) = findmin(abs.(ds .- d))[2]

function pair_video(name, model, glorys, gdays, colormap, colorrange, label)
    fig = Figure(size = (1100, 560))
    title = Observable("")
    left = Observable(masked(model[1]))
    right = Observable(masked(glorys(nearest(gdays, tdays[1]))))
    ax1 = Axis(fig[1, 1], title = title, aspect = DataAspect()); hm = heatmap!(ax1, left; colormap, colorrange, nan_color = :gray80)
    ax2 = Axis(fig[1, 2], title = "GLORYS", aspect = DataAspect()); heatmap!(ax2, right; colormap, colorrange, nan_color = :gray80)
    hidedecorations!(ax1); hidedecorations!(ax2)
    Colorbar(fig[1, 3], hm, label = label)
    CairoMakie.record(fig, joinpath(videodir, name), eachindex(tdays); framerate = 12) do n
        title[] = "model, $(Dates.format(dates[n], "d u yyyy"))"
        left[] = masked(model[n])
        right[] = masked(glorys(nearest(gdays, tdays[n])))
    end
end

pair_video("year_ice.mp4", [clean.(a) for a in he], n -> Float32.(ghe[:, :, n]), gdays, Reverse(:blues), (0, 3), "ice volume per area [m]")
pair_video("year_sst.mp4", T, n -> Float32.(gT[:, :, n]), sdays, :thermal, (-2, 12), "SST [°C]")
pair_video("year_sss.mp4", S, n -> Float32.(gS[:, :, n]), sdays, :haline, (28, 35.5), "SSS [psu]")

fig = Figure(size = (640, 560))
title = Observable("")
frame = Observable(masked(U[1]))
ax = Axis(fig[1, 1], title = title, aspect = DataAspect()); hm = heatmap!(ax, frame; colormap = :viridis, colorrange = (0, 0.6), nan_color = :gray80)
hidedecorations!(ax)
Colorbar(fig[1, 2], hm, label = "surface speed [m s⁻¹]")
CairoMakie.record(fig, joinpath(videodir, "year_speed.mp4"), eachindex(tdays); framerate = 12) do n
    title[] = "surface speed, $(Dates.format(dates[n], "d u yyyy"))"
    frame[] = masked(U[n])
end
@info "Videos written to $videodir"
