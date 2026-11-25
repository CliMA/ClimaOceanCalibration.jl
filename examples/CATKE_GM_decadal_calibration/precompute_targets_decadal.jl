# Targets of the decadal calibration, saved to `calibration_data/decadal_targets.jld2`: temperature and salinity of
# the World Ocean Atlas 2023 climatology of 2005–2014, and the mixed-layer depth climatology of de Boyer Montégut
# (2023) with the spread of its 2008 version.
#
#   julia --project examples/CATKE_GM_decadal_calibration/precompute_targets_decadal.jl

using Downloads
using GibbsSeaWater: gsw_ct_from_t, gsw_sa_from_sp
using NCDatasets

include("data_processing_decadal.jl")

const data_directory = mkpath(joinpath(@__DIR__, "calibration_data"))
const longitudes = -179.5:179.5

# Structural error of temperature [ᵒC], of salinity [g kg⁻¹] and of the logarithm of the mixed-layer depth
const structural_errors = (tropical_T = 0.3, tropical_S = 0.05,
                          mixed_layer_maximum = 0.10, mixed_layer_minimum = 0.11,
                          southern_T = 0.2, southern_S = 0.03)

function cached(filename, url)
    path = joinpath(data_directory, filename)
    isfile(path) || Downloads.download(url, path)
    return path
end

read_variable(path, name, fill) = NCDataset(dataset -> Float64.(coalesce.(Array(dataset[name]), fill)), path)

"""
    woa_record(period)

Conservative Temperature `T` [ᵒC] and Absolute Salinity `S` [g kg⁻¹] of the World Ocean Atlas 2023 climatology of
2005–2014 for `period` ("00" for the year, "01" to "12" for the months), with their standard errors, their numbers of
observations and the `depth_faces` of the levels.
"""
function woa_record(period)
    path(variable) = cached("woa23_A5B4_$(first(variable))$(period)_01.nc",
                            "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/$variable/netcdf/A5B4/1.00/woa23_A5B4_$(first(variable))$(period)_01.nc")

    field(variable, statistic, fill) = read_variable(path(variable), first(variable) * "_" * statistic, fill)[:, :, :, 1]

    t = field("temperature", "an", NaN)
    Sᴾ = field("salinity", "an", NaN)
    depths = read_variable(path("temperature"), "depth", NaN)
    bounds = read_variable(path("temperature"), "depth_bnds", NaN)

    T = fill(NaN, size(t))
    S = fill(NaN, size(t))

    for index in findall(isfinite.(t) .& isfinite.(Sᴾ))
        i, j, k = Tuple(index)
        S[index] = gsw_sa_from_sp(Sᴾ[index], depths[k], longitudes[i], latitudes[j])
        T[index] = gsw_ct_from_t(S[index], t[index], depths[k])
    end

    return (; T, S, depth_faces = [bounds[1, 1]; bounds[2, :]],
              T_error = field("temperature", "se", NaN), S_error = field("salinity", "se", NaN),
              T_observations = field("temperature", "dd", 0), S_observations = field("salinity", "dd", 0))
end

"""
    tracer_target(values, errors, observations, depth_faces, block, model_coverage, structural_error)

The target of the 1ᵒ `values` of each month over `block`: the cells `valid` where the model and the climatology each
cover half of a layer, the `mask` of the boxes in which half of those cells have `observations`, and the values `y`
and the standard errors `observation_error` there. Cells with one observation have no standard error, and boxes
without one are left out.
"""
function tracer_target(values, errors, observations, depth_faces, block, model_coverage, structural_error)
    coverage = last(layer_average(first(values), depth_faces, block.layers))
    valid = @. (model_coverage ≥ 0.5) & (coverage ≥ 0.5)
    observed = [@.(ifelse(isfinite(c), Float64(n > 0), NaN)) for (c, n) in zip(values, observations)]
    errors = [@.(ifelse(n > 1, e, NaN)) for (e, n) in zip(errors, observations)]
    observation_error = block_values(errors, depth_faces, block, valid)
    mask = (block_values(observed, depth_faces, block, valid) .≥ 0.5) .& isfinite.(observation_error)

    return (; y = block_values(values, depth_faces, block, valid)[mask], observation_error = observation_error[mask],
              structural_error, mask, valid)
end

"""
    mixed_layer_targets(model_wet)

The targets of the logarithms of the largest and of the smallest monthly mixed-layer depth in 2ᵒ boxes, over the 1ᵒ
cells where `model_wet` and the climatology are finite. The error of each is that of a median of `n` profiles,
1.25 × 1.48 `MAD` / (`h` √`n`), in the month of the extreme. The boxes kept have observations in every month and an
error.
"""
function mixed_layer_targets(model_wet)
    path = joinpath(data_directory, "mld_dr003_ref10m_v2023.nc")
    isfile(path) || run(`tar -xf $(Downloads.download("https://www.seanoe.org/data/00806/91774/data/103667.tar")) -C $data_directory`)
    h = read_variable(path, "mld_dr003", NaN)
    n = read_variable(path, "n_obs", 0)

    # The 2008 climatology is on 2ᵒ boxes centered on even longitudes and latitudes, with 10⁹ where it has no value
    spread = read_variable(cached("mld_DR003_c1m_reg2.0.nc", "https://mld.ifremer.fr/data/mld_DR003_c1m_reg2.0.nc"), "med_dev", NaN)
    spread[spread .> 1e8] .= NaN
    MAD = [1 ≤ round(Int, φ / 2) + 45 ≤ 89 ? spread[mod(round(Int, λ / 2), 180) + 1, round(Int, φ / 2) + 45, month] : NaN
           for λ in longitudes, φ in latitudes, month in 1:12]

    valid = isfinite.(model_wet) .& dropdims(all(isfinite, h; dims = 3); dims = 3)
    monthly(a) = stack(first(box_average(c, valid .& isfinite.(c), mixed_layer_box)) for c in eachslice(a; dims = 3))

    observed = dropdims(all(monthly(n) .> 0; dims = 3); dims = 3)
    error = monthly(@. 1.25 * 1.48 * MAD / (h * sqrt(n)))
    deepest, shallowest = mixed_layer_extremes(eachslice(h; dims = 3), valid)

    function target(y, month, structural_error)
        observation_error = error[dropdims(month; dims = 3)]
        mask = observed .& isfinite.(observation_error)
        return (; y = y[mask], observation_error = observation_error[mask], structural_error, mask, valid)
    end

    return (; mixed_layer_maximum = target(deepest, last(findmax(monthly(h); dims = 3)), structural_errors.mixed_layer_maximum),
              mixed_layer_minimum = target(shallowest, last(findmin(monthly(h); dims = 3)), structural_errors.mixed_layer_minimum))
end

grid = omip_grid(OneDegree(), CPU())
wet = wet_cells(grid, 1:size(grid, 3))
model_wet = reverse(regrid_to_degrees(ones(size(wet)), wet, degree_regridder(grid)); dims = 3)
model_faces = -reverse(znodes(grid, Face()))

months = [woa_record(lpad(month, 2, '0')) for month in 1:12]
year = woa_record("00")

tropical_coverage = last(layer_average(model_wet, model_faces, tropical.layers))
southern_coverage = last(layer_average(model_wet, model_faces, southern.layers))
monthly(name) = getproperty.(months, name)
depth_faces = first(months).depth_faces

targets = (; tropical_T = tracer_target(monthly(:T), monthly(:T_error), monthly(:T_observations), depth_faces,
                                        tropical, tropical_coverage, structural_errors.tropical_T),
             tropical_S = tracer_target(monthly(:S), monthly(:S_error), monthly(:S_observations), depth_faces,
                                        tropical, tropical_coverage, structural_errors.tropical_S),
             mixed_layer_targets(model_wet[:, :, 1])...,
             southern_T = tracer_target([year.T], [year.T_error], [year.T_observations], year.depth_faces,
                                        southern, southern_coverage, structural_errors.southern_T),
             southern_S = tracer_target([year.S], [year.S_error], [year.S_observations], year.depth_faces,
                                        southern, southern_coverage, structural_errors.southern_S))

for (name, field) in pairs(targets)
    @info "$name: $(length(field.y)) values" extrema(field.y) extrema(field.observation_error)
end

jldsave(joinpath(data_directory, "decadal_targets.jld2"); targets)
