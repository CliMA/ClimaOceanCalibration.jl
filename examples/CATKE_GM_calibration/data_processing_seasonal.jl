# data_processing_seasonal.jl
# Observation/model extraction for the SEASONAL-CYCLE ORCA calibration.
#
# Unlike data_processing.jl (which slices T/S directly on the ORCA grid with no
# regridding), the seasonal target is the *zonal-mean seasonal cycle*: the last
# 12 monthly snapshots, each conservatively regridded ORCA → 1° lat-lon, averaged
# per latitude row, then sliced to the tropics (|lat| ≤ 20°) and upper 200 m. The
# observation vector concatenates, month by month, [T_zonal..., S_zonal...].
#
# Both the model output and the WOA-monthly target go through the SAME pipeline
# (see precompute_woa_monthly_zonal.jl), so the two vectors are comparable.

using Oceananigans
using Oceananigans.Grids: φnodes, znodes, λnodes
using Oceananigans.Fields: Field, CenterField, interior
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy, buoyancy_perturbationᶜᶜᶜ
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using ConservativeRegridding
using JLD2
using LinearAlgebra
using Statistics
using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations: build_grid

const SEASONAL_LAT_RANGE = (-20.0, 20.0)
const SEASONAL_Z_MIN     = -200.0
const SEASONAL_N_MONTHS   = 12

# Shared 1° lat-lon target grid (matches the visualization's ZONAL grid).
const SEASONAL_NLON = 360
const SEASONAL_NLAT = 180
const SEASONAL_LATLON_GRID = LatitudeLongitudeGrid(CPU();
    size      = (SEASONAL_NLON, SEASONAL_NLAT, 1),
    longitude = (0, 360), latitude = (-90, 90), z = (0, 1))
const SEASONAL_LATLON_DST = Field{Center, Center, Nothing}(SEASONAL_LATLON_GRID)

seasonal_zonal_latitudes() = collect(φnodes(SEASONAL_LATLON_GRID, Center()))

"""
    seasonal_regridder(grid) -> ConservativeRegridding.Regridder

Conservative ORCA → 1° lat-lon regridder (single horizontal level). Built once
per grid; the caller should cache it across the 12 months / both tracers.
"""
function seasonal_regridder(grid)
    src = Field{Center, Center, Nothing}(grid)
    return ConservativeRegridding.Regridder(SEASONAL_LATLON_DST, src)
end

"""
    seasonal_ocean_mask_3d(grid) -> Array{Float64,3}

1 in ocean cells, 0 in immersed (land/bathymetry) cells. Same construction as
the visualizer's `build_ocean_mask_3d`.
"""
function seasonal_ocean_mask_3d(grid)
    Nx, Ny, Nz = size(grid)
    mask = ones(Nx, Ny, Nz)
    if grid isa ImmersedBoundaryGrid
        bh = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
        zc = znodes(grid, Center())
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            zc[k] < bh[i, j] && (mask[i, j, k] = 0.0)
        end
    end
    return mask
end

"""
    seasonal_zonal_mean(data_3d, mask_3d, regridder) -> Matrix (Nlat, Nz)

Conservatively regrid each depth level of `data_3d` (weighted by `mask_3d`) onto
the 1° lat-lon grid, then ocean-area-weighted-average per latitude row. NaN where
a latitude row has no ocean coverage at that depth. Mirrors the visualizer's
`compute_zonal_mean`.
"""
function seasonal_zonal_mean(data_3d, mask_3d, regridder)
    Nz    = size(data_3d, 3)
    zonal = fill(NaN, SEASONAL_NLAT, Nz)
    fdata = zeros(SEASONAL_NLON * SEASONAL_NLAT)
    fmask = zeros(SEASONAL_NLON * SEASONAL_NLAT)
    A     = regridder.dst_areas
    for k in 1:Nz
        ConservativeRegridding.regrid!(fdata, regridder, vec(data_3d[:, :, k] .* mask_3d[:, :, k]))
        ConservativeRegridding.regrid!(fmask, regridder, vec(mask_3d[:, :, k]))
        Wd = reshape(fdata .* A, SEASONAL_NLON, SEASONAL_NLAT)
        Wm = reshape(fmask .* A, SEASONAL_NLON, SEASONAL_NLAT)
        for j in 1:SEASONAL_NLAT
            m = sum(@view Wm[:, j])
            m > 0 && (zonal[j, k] = sum(@view Wd[:, j]) / m)
        end
    end
    return zonal
end

"""
    slice_tropics_top(zonal, latitude, depth; lat_range, z_min) -> Vector{Float64}

Flatten the tropical (|lat| ≤ lat_range), upper-`|z_min|` m portion of a
`(Nlat, Nz)` zonal-mean array, dropping NaN (rows/depths with no ocean).
"""
function slice_tropics_top(zonal, latitude, depth;
                           lat_range = SEASONAL_LAT_RANGE, z_min = SEASONAL_Z_MIN)
    j_idx = findall(φ -> lat_range[1] <= φ <= lat_range[2], latitude)
    k_idx = findall(z -> z >= z_min, depth)
    sub   = zonal[j_idx, k_idx]
    return Vector{Float64}(filter(isfinite, vec(sub)))
end

"""
    load_member_monthly_TS(member_dir, filename_prefix) -> (Ts, Ss, grid, times)

Load the final 12 monthly T,S snapshots from `<prefix>_monthly_TS.jld2` as
`(Nx,Ny,Nz)` arrays, with their snapshot `times` (seconds). Errors if fewer
than 12 monthly snapshots were written.
"""
function load_member_monthly_TS(member_dir::AbstractString, filename_prefix::AbstractString)
    path = joinpath(member_dir, "$(filename_prefix)_monthly_TS.jld2")
    isfile(path) || error("load_member_monthly_TS: missing $path (seasonal forward model output)")
    Tfts = FieldTimeSeries(path, "T")
    Sfts = FieldTimeSeries(path, "S")
    nt = length(Tfts.times)
    nt >= SEASONAL_N_MONTHS ||
        error("load_member_monthly_TS: only $nt monthly snapshots in $path; need $SEASONAL_N_MONTHS (≥ 1 yr)")
    last12 = (nt - SEASONAL_N_MONTHS + 1):nt
    grid = Tfts.grid
    Ts = [Array(interior(Tfts[n])) for n in last12]
    Ss = [Array(interior(Sfts[n])) for n in last12]
    times = collect(Tfts.times[last12])
    return Ts, Ss, grid, times
end

"""
    seasonal_zonal_TS_2d(Ts, Ss, grid) -> (zonal_T, zonal_S, latitude, depth)

Regrid + zonal-average the monthly T,S 3-D arrays. Returns `zonal_T` / `zonal_S`
as length-N vectors of `(Nlat, Nz)` matrices (full latitude/depth, *un-sliced*),
plus the lat-lon latitude axis and the model depth axis. The single regridding
pass that both the observation vector and the diagnostic videos share.
"""
function seasonal_zonal_TS_2d(Ts, Ss, grid)
    rg       = seasonal_regridder(grid)
    mask     = seasonal_ocean_mask_3d(grid)
    latitude = seasonal_zonal_latitudes()
    depth    = collect(znodes(grid, Center()))
    zonal_T  = [seasonal_zonal_mean(Ts[m], mask, rg) for m in 1:length(Ts)]
    zonal_S  = [seasonal_zonal_mean(Ss[m], mask, rg) for m in 1:length(Ss)]
    return zonal_T, zonal_S, latitude, depth
end

"""
    seasonal_buoyancy_3d(T::Field, S::Field) -> Array{Float64,3}

Buoyancy perturbation from TEOS-10 T,S `CenterField`s, using the same
`SeawaterBuoyancy` formulation (default gravity + reference density) as the OMIP
model's `bo` output. Used to build the WOA buoyancy reference in the precompute.
"""
function seasonal_buoyancy_3d(T::Field, S::Field)
    grid = T.grid
    buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())
    bop = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy, (T = T, S = S))
    B = Field(bop)
    compute!(B)
    return Array(interior(B))
end

"""
    member_zonal_TSB_2d(member_dir, filename_prefix) -> (zT, zS, zb, latitude, depth, times)

Load a member's final 12 monthly T, S, and buoyancy snapshots and return their
zonal-mean `(Nlat, Nz)` arrays (12-vectors each), plus axes and snapshot times.
Builds the regridder once and reuses it for all three fields. Used by the
per-member seasonal video.
"""
function member_zonal_TSB_2d(member_dir::AbstractString, filename_prefix::AbstractString)
    path = joinpath(member_dir, "$(filename_prefix)_monthly_TS.jld2")
    isfile(path) || error("member_zonal_TSB_2d: missing $path")
    Tfts = FieldTimeSeries(path, "T")
    Sfts = FieldTimeSeries(path, "S")
    Bfts = FieldTimeSeries(path, "bo")
    nt = length(Tfts.times)
    nt >= SEASONAL_N_MONTHS ||
        error("member_zonal_TSB_2d: only $nt monthly snapshots in $path; need $SEASONAL_N_MONTHS")
    last12   = (nt - SEASONAL_N_MONTHS + 1):nt
    grid     = Tfts.grid
    rg       = seasonal_regridder(grid)
    mask     = seasonal_ocean_mask_3d(grid)
    latitude = seasonal_zonal_latitudes()
    depth    = collect(znodes(grid, Center()))
    zT = [seasonal_zonal_mean(Array(interior(Tfts[n])), mask, rg) for n in last12]
    zS = [seasonal_zonal_mean(Array(interior(Sfts[n])), mask, rg) for n in last12]
    zb = [seasonal_zonal_mean(Array(interior(Bfts[n])), mask, rg) for n in last12]
    times = collect(Tfts.times[last12])
    return zT, zS, zb, latitude, depth, times
end

"""
    flatten_seasonal_zonal(zonal_T, zonal_S, latitude, depth; lat_range, z_min) -> Vector

Concatenate the 2-D zonal arrays month-by-month into the observation vector:
for each month, [T_zonal_tropics_top..., S_zonal_tropics_top...].
"""
function flatten_seasonal_zonal(zonal_T, zonal_S, latitude, depth;
                                lat_range = SEASONAL_LAT_RANGE, z_min = SEASONAL_Z_MIN)
    out = Float64[]
    for m in 1:length(zonal_T)
        append!(out, slice_tropics_top(zonal_T[m], latitude, depth; lat_range, z_min))
        append!(out, slice_tropics_top(zonal_S[m], latitude, depth; lat_range, z_min))
    end
    return out
end

"""
    seasonal_zonal_TS_vector(Ts, Ss, grid; lat_range, z_min) -> Vector{Float64}

Given monthly T,S 3-D arrays on `grid`, build the concatenated seasonal
observation vector. Thin wrapper over `seasonal_zonal_TS_2d` + `flatten_seasonal_zonal`.
"""
function seasonal_zonal_TS_vector(Ts, Ss, grid;
                                  lat_range = SEASONAL_LAT_RANGE, z_min = SEASONAL_Z_MIN)
    zonal_T, zonal_S, latitude, depth = seasonal_zonal_TS_2d(Ts, Ss, grid)
    return flatten_seasonal_zonal(zonal_T, zonal_S, latitude, depth; lat_range, z_min)
end

"""
    process_member_data_seasonal(member_dir, filename_prefix; lat_range, z_min)

Read a member's monthly T,S, regrid + zonal-average + slice, and return the
concatenated 12-month [T..., S...] seasonal observation vector.
"""
function process_member_data_seasonal(member_dir::AbstractString,
                                      filename_prefix::AbstractString = "orca_calib";
                                      lat_range = SEASONAL_LAT_RANGE,
                                      z_min     = SEASONAL_Z_MIN)
    Ts, Ss, grid, _ = load_member_monthly_TS(member_dir, filename_prefix)
    return seasonal_zonal_TS_vector(Ts, Ss, grid; lat_range, z_min)
end

"""
    member_has_seasonal_output(member_dir, filename_prefix) -> Bool

True if the member wrote the monthly-TS seasonal file.
"""
function member_has_seasonal_output(member_dir::AbstractString, filename_prefix::AbstractString = "orca_calib")
    return isfile(joinpath(member_dir, "$(filename_prefix)_monthly_TS.jld2"))
end

"""
    load_woa_seasonal_target(woa_seasonal_file) -> Vector{Float64}

Load the precomputed WOA-monthly zonal seasonal target produced by
precompute_woa_monthly_zonal.jl (already sliced to the same tropics/depth and
laid out month-by-month [T..., S...]).
"""
function load_woa_seasonal_target(woa_seasonal_file::AbstractString)
    isfile(woa_seasonal_file) || error("""
        WOA seasonal-zonal target not found at:
            $woa_seasonal_file
        Run precompute_woa_monthly_zonal.jl first.
    """)
    return jldopen(woa_seasonal_file, "r") do file
        Vector{Float64}(file["Y_target"])
    end
end

"""
    build_seasonal_covariance(Y_target; T_variance, S_variance, n_months) -> Diagonal

Diagonal observation covariance for the month-by-month [T..., S...] layout. Each
month contributes an equal-length T block then S block; the total length must be
divisible by `2 * n_months`.
"""
function build_seasonal_covariance(Y_target::AbstractVector;
                                   T_variance::Real = 0.2^2,
                                   S_variance::Real = (0.2 / 4)^2,
                                   n_months::Int    = SEASONAL_N_MONTHS)
    n = length(Y_target)
    n % (2 * n_months) == 0 ||
        error("build_seasonal_covariance: length $n not divisible by 2*n_months=$(2*n_months)")
    L = n ÷ (2 * n_months)
    diagv = Float64[]
    for _ in 1:n_months
        append!(diagv, fill(float(T_variance), L))
        append!(diagv, fill(float(S_variance), L))
    end
    return Diagonal(diagv)
end
