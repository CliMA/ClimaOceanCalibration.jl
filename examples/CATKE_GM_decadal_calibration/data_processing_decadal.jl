# Reduction of monthly temperature, salinity and mixed-layer depth to the targets of the decadal calibration. Model
# output and observations take the same steps from the 1ᵒ latitude-longitude grid of the observations onward: layer
# means, box means over the cells that both cover, and the mask of the target.

using ClimaOcean: OneDegree, omip_grid
using ConservativeRegridding
using JLD2
using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: inactive_node
using Statistics

const degree_grid = LatitudeLongitudeGrid(CPU(); size = (360, 180, 1), longitude = (-180, 180), latitude = (-90, 90), z = (0, 1))
const latitudes = -89.5:89.5

# Layer edges [m], the 1ᵒ cells in longitude and latitude of each box, and the rows of boxes kept:
# 2ᵒ boxes of 20ᵒS–20ᵒN to 300 m, and zonal means of 65ᵒS–40ᵒS to 2000 m
const tropical = (layers = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300], box = (2, 2), rows = 36:55)
const southern = (layers = [0:50:500; 750:250:2000], box = (360, 1), rows = 26:50)
const mixed_layer_box = (2, 2)

"""
    layer_average(values, depth_faces, layers)

Thickness-weighted mean of the finite `values[i, j, k]`, whose cell `k` lies between the depths `depth_faces[k]` and
`depth_faces[k + 1]`, over each layer between consecutive edges of `layers`, and the fraction of each layer that the
finite values cover.
"""
function layer_average(values, depth_faces, layers)
    Nx, Ny, Nz = size(values)
    Nl = length(layers) - 1
    sums = zeros(Nx, Ny, Nl)
    thickness = zeros(Nx, Ny, Nl)

    for l in 1:Nl, k in 1:Nz
        overlap = min(depth_faces[k + 1], layers[l + 1]) - max(depth_faces[k], layers[l])
        overlap > 0 || continue
        for j in 1:Ny, i in 1:Nx
            if isfinite(values[i, j, k])
                sums[i, j, l] += overlap * values[i, j, k]
                thickness[i, j, l] += overlap
            end
        end
    end

    return sums ./ thickness, thickness ./ reshape(diff(layers), 1, 1, Nl)
end

"""
    box_average(values, weights, box)

Mean of `values[i, j, ...]` on the 1ᵒ grid over boxes of `box[1]` by `box[2]` cells, weighted by `weights` and the
cell area, and the fraction of the area of each box that carries weight.
"""
function box_average(values, weights, box)
    area = cosd.(latitudes)'
    boxes(a) = dropdims(sum(reshape(a, box[1], 360 ÷ box[1], box[2], 180 ÷ box[2], size(a)[3:end]...); dims = (1, 3)); dims = (1, 3))
    total = boxes(weights .* area)
    return boxes(@. ifelse(weights > 0, values, 0) * weights * area) ./ total, total ./ boxes(area .* one.(weights))
end

"""
    block_values(monthly, depth_faces, block, valid)

The means of each of the `monthly` 1ᵒ fields over the layers and boxes of `block`, over the finite cells of `valid`, as an
array of longitude, latitude, layer and month.
"""
function block_values(monthly, depth_faces, block, valid)
    layered = [first(layer_average(c, depth_faces, block.layers)) for c in monthly]
    return stack(first(box_average(c, valid .& isfinite.(c), block.box))[:, block.rows, :] for c in layered)
end

"""
    mixed_layer_extremes(monthly, valid)

The logarithms of the largest and of the smallest of the `monthly` mixed-layer depths, averaged over 2ᵒ boxes of the
cells of `valid`.
"""
function mixed_layer_extremes(monthly, valid)
    boxes = stack(first(box_average(h, valid, mixed_layer_box)) for h in monthly)
    return log.(dropdims(maximum(boxes; dims = 3); dims = 3)), log.(dropdims(minimum(boxes; dims = 3); dims = 3))
end

"""
    degree_regridder(grid)

The conservative regridder from the cell centers of `grid` to the 1ᵒ grid.
"""
degree_regridder(grid) = ConservativeRegridding.Regridder(Field{Center, Center, Nothing}(degree_grid),
                                                          Field{Center, Center, Nothing}(grid))

"""
    regrid_to_degrees(values, wet, regridder)

`values[i, j, k]` on the model grid regridded level by level to the 1ᵒ grid over the `wet` cells, NaN where less than
half of a 1ᵒ cell is wet.
"""
function regrid_to_degrees(values, wet, regridder)
    regridded = zeros(360, 180, size(values, 3))
    sums = zeros(360 * 180)
    wet_fraction = zeros(360 * 180)

    for k in axes(values, 3)
        ConservativeRegridding.regrid!(sums, regridder, vec(values[:, :, k] .* wet[:, :, k]))
        ConservativeRegridding.regrid!(wet_fraction, regridder, vec(Float64.(wet[:, :, k])))
        regridded[:, :, k] .= reshape(@.(ifelse(wet_fraction ≥ 0.5, sums / wet_fraction, NaN)), 360, 180)
    end

    return regridded
end

"""
    wet_cells(grid, levels)

Whether each cell center of the `levels` of `grid` is in the ocean.
"""
wet_cells(grid, levels) = [!inactive_node(i, j, k, grid, Center(), Center(), Center())
                           for i in 1:size(grid, 1), j in 1:size(grid, 2), k in levels]

"""
    monthly_climatology(path, name, years)

The mean of each of the 12 months of the year over the last `years` of monthly records of `name` in the output file
at `path`.
"""
function monthly_climatology(path, name, years)
    return jldopen(path) do file
        iterations = sort(parse.(Int, keys(file["timeseries/t"])))
        records = reshape(iterations[end - 12years + 1:end], 12, years)
        [mean(Float64.(file["timeseries/$name/$iteration"]) for iteration in records[month, :]) for month in 1:12]
    end
end

"""
    member_observations(directory, targets; years = 10)

The vector that the last `years` of monthly output of the member in `directory` give for `targets`, in the order of
its fields.
"""
function member_observations(directory, targets; years = 10)
    grid = omip_grid(OneDegree(), CPU())
    regridder = degree_regridder(grid)
    Nz = size(grid, 3)

    T = monthly_climatology(joinpath(directory, "monthly_tracers.jld2"), "T", years)
    S = monthly_climatology(joinpath(directory, "monthly_tracers.jld2"), "S", years)
    h = monthly_climatology(joinpath(directory, "monthly_mixed_layer_depth.jld2"), "h", years)

    levels = Nz - size(first(T), 3) + 1:Nz
    wet = wet_cells(grid, levels)
    depth_faces = -reverse(znodes(grid, Face())[first(levels):Nz + 1])

    T = [reverse(regrid_to_degrees(c, wet, regridder); dims = 3) for c in T]
    S = [reverse(regrid_to_degrees(c, wet, regridder); dims = 3) for c in S]
    h = [dropdims(regrid_to_degrees(c, wet[:, :, end:end], regridder); dims = 3) for c in h]

    deepest, shallowest = mixed_layer_extremes(h, targets.mixed_layer_maximum.valid)

    return vcat(block_values(T, depth_faces, tropical, targets.tropical_T.valid)[targets.tropical_T.mask],
                block_values(S, depth_faces, tropical, targets.tropical_S.valid)[targets.tropical_S.mask],
                deepest[targets.mixed_layer_maximum.mask],
                shallowest[targets.mixed_layer_minimum.mask],
                block_values([mean(T)], depth_faces, southern, targets.southern_T.valid)[targets.southern_T.mask],
                block_values([mean(S)], depth_faces, southern, targets.southern_S.valid)[targets.southern_S.mask])
end

"""
    field_ranges(targets)

The range of each field of `targets` in the observation vector.
"""
function field_ranges(targets)
    stops = cumsum([length(field.y) for field in targets])
    return NamedTuple{keys(targets)}(Tuple(stop - length(field.y) + 1:stop for (stop, field) in zip(stops, targets)))
end
