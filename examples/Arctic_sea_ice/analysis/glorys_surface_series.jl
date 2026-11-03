# GLORYS daily surface temperature and salinity on a run's grid, for comparison with the run's daily surface output.
# Usage: julia --project=.. glorys_surface_series.jl <run dir> <start date> <ndays> [<output file> <run start date> <GLORYS cache>]
# The optional arguments let a long record be split over jobs (each file counts time from the run start date) and point at another cache.
# Reads the surface level of the inpainted GLORYS fields in the cache (as left by a run or prefetch_glorys.jl),
# interpolates it onto the grid of the run's arctic_ocean_surface.jld2 and writes glorys_surface_series.jld2
# next to it: times [s] and the daily surface T [°C] and S [psu] fields.

using NumericalEarth, Oceananigans, Oceananigans.Units, Dates, CopernicusMarine
using NumericalEarth.DataWrangling: native_grid, inpainted_metadata_path
using Oceananigans.Fields: interpolate!
const JLD2 = Oceananigans.OutputWriters.JLD2

dir   = ARGS[1]
day₀  = DateTime(ARGS[2])
dates = day₀ : Day(1) : day₀ + Day(parse(Int, ARGS[3]))
output = joinpath(dir, length(ARGS) >= 4 ? ARGS[4] : "glorys_surface_series.jld2")
glorys_dir = expanduser(length(ARGS) >= 6 ? ARGS[6] : "~/GLORYS_data")

run_surface = FieldTimeSeries(joinpath(dir, "arctic_ocean_surface.jld2"), "T"; architecture = CPU(), backend = OnDisk())
grid = run_surface.grid
dataset = GLORYSDaily()
region  = BoundingBox(longitude = (-180, 180), latitude = (43, 90))

halo = (3, 3, 3)
function surface(name, date)
    metadatum = Metadatum(name; date, dataset, region, dir = glorys_dir)
    native = native_grid(metadatum, CPU(); halo)
    data = JLD2.jldopen(f -> f["data"], inpainted_metadata_path(metadatum))
    source = Field{Center, Center, Nothing}(native)
    interior(source, :, :, 1) .= data[halo[1]+1:end-halo[1], halo[2]+1:end-halo[2], size(native, 3)+halo[3]]
    target = Field{Center, Center, Nothing}(grid)
    interpolate!(target, source)
    return Array(interior(target, :, :, 1))
end

Ts = [surface(:temperature, d) for d in dates]
Ss = [surface(:salinity, d) for d in dates]
run_start = DateTime(length(ARGS) >= 5 ? ARGS[5] : ARGS[2])
times = [(d - run_start) / Millisecond(1000) + 12hours for d in dates]   # daily means sit at noon, counted from the run's start date

JLD2.jldsave(output; times, T = cat(Ts..., dims = 3), S = cat(Ss..., dims = 3))
@info "wrote $output for $(length(dates)) days"
