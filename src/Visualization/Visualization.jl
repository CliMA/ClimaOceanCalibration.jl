module Visualization

export Run, dashboard

using Dates
using Makie
using JLD2
using NaNStatistics: nanmean
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.Grids: nodes, OrthogonalSphericalShellGrid
using Oceananigans.OutputReaders: FieldTimeSeries, OnDisk, InMemory
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: prettytime
using Oceananigans.Units: days
using ..OMIPSimulations: jld2_output_part_paths

#####
##### Runs
#####

"""
    Run(dir; name = basename(dir))
    Run(name, series)

A named collection of `FieldTimeSeries`. Reading `dir` collects every variable of every
Oceananigans JLD2 output in it (split `_partN` files included) under the key
`"group/variable"`, where `group` is the file stem without the prefix shared by all files.
Series with at most one spatial dimension are held in memory; the rest are read from disk
on demand.
"""
struct Run
    name :: String
    series :: Dict{String, FieldTimeSeries}
end

Run(name::AbstractString, series::AbstractDict) = Run(String(name), Dict{String, FieldTimeSeries}(series))

function Run(dir::AbstractString; name = basename(rstrip(dir, '/')))
    files = filter(f -> endswith(f, ".jld2") && !occursin("checkpoint", f), readdir(dir))
    stems = unique!(replace.(files, r"(_part\d+)?\.jld2$" => ""))
    prefix = common_prefix(stems)
    series = Dict{String, FieldTimeSeries}()
    for stem in stems
        path = joinpath(dir, stem * ".jld2")
        group = chopprefix(stem, prefix)
        grid = nothing
        for variable in variables(first(jld2_output_part_paths(path)))
            fts = FieldTimeSeries(path, variable; backend = OnDisk(), grid)
            grid = fts.grid
            length(spatial_dims(fts)) ≤ 1 && (fts = FieldTimeSeries(path, variable; backend = InMemory(), grid))
            series[group * "/" * variable] = fts
        end
    end
    return Run(name, series)
end

variables(path) = jldopen(path) do file
    haskey(file, "timeseries") ? filter(!=("t"), keys(file["timeseries"])) : String[]
end

function common_prefix(stems)
    length(stems) > 1 || return ""
    chars = collect.(stems)
    n = 0
    while all(c -> length(c) > n && c[n + 1] == chars[1][n + 1], chars)
        n += 1
    end
    prefix = join(chars[1][1:n])
    cut = findlast('_', prefix)
    return isnothing(cut) ? "" : prefix[1:cut]
end

spatial_dims(fts) = findall(>(1), size(fts)[1:3])

Base.keys(run::Run) = sort!(collect(keys(run.series)))
Base.getindex(run::Run, key::AbstractString) = run.series[key]
common_keys(runs) = mapreduce(keys, intersect, runs)

Base.show(io::IO, run::Run) = print(io, "Run(\"", run.name, "\", ", length(run.series), " series)")

function Base.show(io::IO, ::MIME"text/plain", run::Run)
    print(io, "Run \"", run.name, "\" with ", length(run.series), " series")
    for key in keys(run)
        print(io, "\n  ", rpad(key, 32), join(size(run[key]), "×"))
    end
end

#####
##### Snapshots and sections
#####

function snapshot(fts, t)
    field = fts[argmin(abs.(fts.times .- t))]
    mask_immersed_field!(field, NaN)
    return Array(interior(field))
end

squeeze(A) = dropdims(A; dims = Tuple(findall(==(1), size(A))))

function plane(A, dim, index, average)
    count(>(1), size(A)) == 3 || return squeeze(A)
    average && return squeeze(nanmean(A; dims = dim))
    return Array(squeeze(selectdim(A, dim, index)))
end

function spatial_axes(fts)
    grid = fts.grid isa ImmersedBoundaryGrid ? fts.grid.underlying_grid : fts.grid
    ℓ = map(L -> L === Nothing ? Center() : L(), location(fts))
    ξ = nodes(grid, ℓ...)
    names = grid isa Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid} ? ("λ", "φ", "z") : ("x", "y", "z")
    coordinates = ntuple(d -> ndims(ξ[d]) == 1 ? ξ[d][fts.indices[d]] : (1:size(fts, d)), 3)
    labels = ntuple(d -> ndims(ξ[d]) == 1 ? names[d] : ("i", "j", "k")[d], 3)
    return coordinates, labels
end

function rescale!(colorrange, planes, transform = identity)
    values = filter(isfinite, reduce(vcat, vec(p[]) for p in planes))
    isempty(values) || (colorrange[] = transform(extrema(values)))
    return nothing
end

symmetric(range) = (-maximum(abs, range), maximum(abs, range))

#####
##### Dashboard
#####

"""
    dashboard(runs...; fields, section = :x, reference_date = nothing, colormap = :viridis)

Interactive figure with one row per field and one column per run. Three-dimensional
fields are shown as sections normal to `section` (`:x`, `:y` or `:z`) at the index set by
the slice slider, or averaged along it when the mean toggle is on; two-dimensional fields
as maps; one-dimensional fields as profiles; scalar series against time. The time slider
selects the nearest snapshot of every series. Two runs on grids of equal size get a
difference column. The menu on every row switches among the variables of the same shape.
"""
function dashboard(runs::Run...; fields = first(common_keys(runs), 3), section = :x,
                   reference_date = nothing, colormap = :viridis)
    dim = (x = 1, y = 2, z = 3)[section]
    all_series = [fts for run in runs for fts in values(run.series)]
    times = sort!(unique!(mapreduce(fts -> collect(fts.times), vcat, all_series)))
    slice_range = 1:maximum(size(fts, dim) for fts in all_series)

    nruns = length(runs)
    nrows = length(fields)
    fig = Figure(size = (240 + 400 * (nruns + (nruns == 2)), 120 + 300 * nrows))

    sliders = SliderGrid(fig[nrows + 1, 2:nruns + 1],
                         (label = "time", range = 1:length(times), startvalue = length(times), update_while_dragging = false,
                          format = n -> isnothing(reference_date) ? prettytime(times[n]) :
                                        Dates.format(reference_date + Second(round(Int, times[n])), "yyyy-mm-dd")),
                         (label = ("i", "j", "k")[dim], range = slice_range, startvalue = cld(last(slice_range), 2)))
    time = @lift times[$(sliders.sliders[1].value)]
    index = sliders.sliders[2].value

    controls = fig[nrows + 2, 2:nruns + 1] = GridLayout(tellwidth = false)
    toggle = Toggle(controls[1, 1])
    Label(controls[1, 2], "mean along $section")
    button = Button(controls[1, 3]; label = "rescale colors")

    rescalers = Function[]
    for (r, key) in enumerate(fields)
        options = filter(k -> all(size(run[k])[1:3] == size(run[key])[1:3] for run in runs), common_keys(runs))
        menu = Menu(fig[r, 1]; options, default = key, width = 180, valign = :top, tellheight = false)
        dimensionality = length(spatial_dims(runs[1][key]))
        if dimensionality == 0
            timeseries_row!(fig[r, 2:nruns + 1], runs, menu.selection, time)
        elseif dimensionality == 1
            profile_row!(fig[r, 2:nruns + 1], runs, menu.selection, time)
        else
            push!(rescalers, heatmap_row!(fig, r, runs, menu.selection, time, dim, index, toggle.active, colormap))
        end
    end
    on(_ -> foreach(f -> f(), rescalers), button.clicks)

    return fig
end

function heatmap_row!(fig, r, runs, field, time, dim, index, average, colormap)
    nruns = length(runs)
    indices = [@lift(clamp($index, 1, size(run[field[]], dim))) for run in runs]
    snapshots = [@lift(snapshot(run[$field], $time)) for run in runs]
    planes = [@lift(plane($snapshot, dim, $index, $average)) for (snapshot, index) in zip(snapshots, indices)]
    colorrange = Observable((0.0, 1.0))
    rescale!(colorrange, planes)
    for (c, run) in enumerate(runs)
        heatmap_panel!(fig[r, 1 + c], run[field[]], planes[c], dim, indices[c], run.name; colormap, colorrange)
    end
    Colorbar(fig[r, nruns + 2]; colormap, limits = colorrange)

    rescale = () -> rescale!(colorrange, planes)
    if nruns == 2 && size(planes[1][]) == size(planes[2][])
        difference = @lift $(planes[1]) .- $(planes[2])
        difference_range = Observable((-1.0, 1.0))
        rescale!(difference_range, (difference,), symmetric)
        heatmap_panel!(fig[r, nruns + 3], runs[1][field[]], difference, dim, indices[1], "$(runs[1].name) − $(runs[2].name)";
                       colormap = :balance, colorrange = difference_range)
        Colorbar(fig[r, nruns + 4]; colormap = :balance, limits = difference_range)
        rescale = () -> (rescale!(colorrange, planes); rescale!(difference_range, (difference,), symmetric))
    end
    on(_ -> rescale(), field)
    return rescale
end

function heatmap_panel!(position, fts, plane, dim, index, title; kwargs...)
    coordinates, labels = spatial_axes(fts)
    shown = length(spatial_dims(fts)) == 3 ? filter(!=(dim), 1:3) : spatial_dims(fts)
    x, y = coordinates[shown]
    ax = Axis(position; title, xlabel = labels[shown[1]], ylabel = labels[shown[2]])
    heatmap!(ax, x, y, plane; nan_color = :lightgray, kwargs...)
    dim == shown[1] && vlines!(ax, @lift([x[$index]]); color = :white, linestyle = :dash)
    dim == shown[2] && hlines!(ax, @lift([y[$index]]); color = :white, linestyle = :dash)
    return nothing
end

function profile_row!(position, runs, field, time)
    d = only(spatial_dims(runs[1][field[]]))
    label = spatial_axes(runs[1][field[]])[2][d]
    name = @lift string($field)
    ax = d == 3 ? Axis(position; xlabel = name, ylabel = label) : Axis(position; xlabel = label, ylabel = name)
    for run in runs
        ξ = spatial_axes(run[field[]])[1][d]
        values = @lift vec(snapshot(run[$field], $time))
        points = d == 3 ? (@lift Point2f.($values, ξ)) : (@lift Point2f.(ξ, $values))
        lines!(ax, points; label = run.name)
        on(_ -> autolimits!(ax), points)
    end
    axislegend(ax)
    return nothing
end

function timeseries_row!(position, runs, field, time)
    ax = Axis(position; xlabel = "time (years)", ylabel = @lift(string($field)))
    for run in runs
        points = @lift let fts = run[$field]
            Point2f.(fts.times ./ 365days, [interior(fts[n])[] for n in eachindex(fts.times)])
        end
        lines!(ax, points; label = run.name)
        on(_ -> autolimits!(ax), points)
    end
    vlines!(ax, @lift([$time / 365days]); color = :black, linestyle = :dash)
    axislegend(ax)
    return nothing
end

end # module
