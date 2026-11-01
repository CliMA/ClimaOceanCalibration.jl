# GLORYS daily sea-ice extent and volume on a run's grid, for comparison with the run's daily ice output.
# Usage: julia --project=.. glorys_ice_series.jl <run dir> <start date> <ndays> [<GLORYS cache, default ~/GLORYS_data>]
# Reads the grid from the run's arctic_sea_ice_surface.jld2 and writes glorys_ice.jld2 next to it:
# times [s], extent [m²] (concentration > 0.15), volume [m³], plus the daily concentration and h·ℵ fields.

using NumericalEarth, Oceananigans, Dates, CopernicusMarine
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_inactive_node
const JLD2 = Oceananigans.OutputWriters.JLD2

dir   = ARGS[1]
day₀  = DateTime(ARGS[2])
dates = day₀ : Day(1) : day₀ + Day(parse(Int, ARGS[3]))

he = FieldTimeSeries(joinpath(dir, "arctic_sea_ice_surface.jld2"), "he"; architecture = CPU(), backend = OnDisk())
grid = he.grid
Nx, Ny, Nz = size(grid)
dataset = GLORYSDaily()
region  = BoundingBox(longitude = (-180, 180), latitude = (43, 90))
glorys_dir = expanduser(length(ARGS) >= 4 ? ARGS[4] : "~/GLORYS_data")

hᵉ = FieldTimeSeries(Metadata(:sea_ice_thickness;     dates, dataset, region, dir = glorys_dir), grid; time_indices_in_memory = length(dates))
ℵᵉ = FieldTimeSeries(Metadata(:sea_ice_concentration; dates, dataset, region, dir = glorys_dir), grid; time_indices_in_memory = length(dates))

wet  = [!immersed_inactive_node(i, j, Nz, grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
area = [Azᶜᶜᶜ(i, j, Nz, grid) for i in 1:Nx, j in 1:Ny]

clean(x) = isnan(x) ? 0.0 : x
n = eachindex(dates)
ℵs = [clamp.(clean.(Array(interior(ℵᵉ[k], :, :, 1))), 0, 1) for k in n]
hs = [max.(clean.(Array(interior(hᵉ[k], :, :, 1))), 0) for k in n]
extent = [sum(area[wet .& (ℵ .> 0.15)]) for ℵ in ℵs]
volume = [sum((h .* ℵ .* area)[wet]) for (h, ℵ) in zip(hs, ℵs)]

JLD2.jldsave(joinpath(dir, "glorys_ice.jld2");
             times = collect(ℵᵉ.times), extent, volume,
             concentration = cat(ℵs..., dims = 3), volume_per_area = cat([h .* ℵ for (h, ℵ) in zip(hs, ℵs)]..., dims = 3))
@info "wrote $(joinpath(dir, "glorys_ice.jld2")): extent $(round(extent[1] / 1e12; digits = 2)) → $(round(extent[end] / 1e12; digits = 2)) million km², volume $(round(volume[1] / 1e12; digits = 2)) → $(round(volume[end] / 1e12; digits = 2)) thousand km³"
