# data_processing.jl
# Tropical-belt observation/model extraction utilities for the ORCA-grid
# CATKE+GM calibration.
#
# Model output and WOA both live on the ORCA grid, so there is no XESMF
# regridding step — we just slice by latitude / depth, drop NaNs, and
# concatenate [T..., S...] into a single observation vector.

using Oceananigans
using Oceananigans.Grids: φnodes, znodes
using Oceananigans.Fields: Field, CenterField, location, interior
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using JLD2
using LinearAlgebra
using Statistics
using NaNStatistics
using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations: build_grid

const DEFAULT_LAT_RANGE = (-20.0, 20.0)
const DEFAULT_Z_MIN     = -200.0

"""
    load_orca_5yr_average(member_dir, filename_prefix) -> (T, S, u)

Load the per-member 5-year time-mean T, S, u as Oceananigans `Field`s.
"""
function load_orca_5yr_average(member_dir::AbstractString, filename_prefix::AbstractString = "orca_calib")
    path = joinpath(member_dir, "$(filename_prefix)_5year_average.jld2")
    T = FieldTimeSeries(path, "T")[end]
    S = FieldTimeSeries(path, "S")[end]
    u = FieldTimeSeries(path, "u")[end]
    return T, S, u
end

"""
    extract_tropics_top(field; lat_range, z_min) -> Vector{Float64}

Extract the tropical, upper-`|z_min|` m section of an Oceananigans `Field`,
drop NaNs (immersed cells / land), and return a flat vector.
"""
function extract_tropics_top(field::Field; lat_range = DEFAULT_LAT_RANGE, z_min = DEFAULT_Z_MIN)
    f   = on_architecture(CPU(), field)
    LX, LY, LZ = location(f)
    grid = f.grid

    φ = φnodes(grid, LX(), LY(), LZ())
    z = znodes(grid, LX(), LY(), LZ())

    lat_indices = findall(φᵢ -> lat_range[1] <= φᵢ <= lat_range[2], φ)
    z_indices   = findall(zᵢ -> zᵢ >= z_min, z)

    mask_immersed_field!(f, NaN)
    data = interior(f, :, lat_indices, z_indices)
    return Vector{Float64}(data[.!isnan.(data)])
end

"""
    process_member_data(member_dir, filename_prefix; lat_range, z_min) -> Vector{Float64}

Read a member's 5-year mean and return the flat `[T..., S...]` observation
vector restricted to the tropics in the top |z_min| m.
"""
function process_member_data(member_dir::AbstractString,
                             filename_prefix::AbstractString = "orca_calib";
                             lat_range = DEFAULT_LAT_RANGE,
                             z_min     = DEFAULT_Z_MIN)
    T, S, _ = load_orca_5yr_average(member_dir, filename_prefix)
    T_vec = extract_tropics_top(T; lat_range, z_min)
    S_vec = extract_tropics_top(S; lat_range, z_min)
    return vcat(T_vec, S_vec)
end

"""
    load_woa_on_orca(woa_file, Nz, depth) -> (T_field, S_field)

Load the precomputed WOA-on-ORCA-grid file produced by
`precompute_woa_orca.jl` and wrap the arrays in `CenterField`s on a freshly
built ORCA grid (CPU). The grid is rebuilt from the same `(Nz, depth)`
parameters that were stored in the file so the field locations line up
with what the model produced.
"""
function load_woa_on_orca(woa_file::AbstractString)
    data = jldopen(woa_file, "r") do file
        return (T = file["T"], S = file["S"], Nz = file["Nz"], depth = file["depth"])
    end

    grid = build_grid(Val(:orca), CPU(), data.Nz, data.depth)
    T = CenterField(grid)
    S = CenterField(grid)
    interior(T) .= data.T
    interior(S) .= data.S
    return T, S
end

"""
    process_woa_target(woa_file; lat_range, z_min) -> Vector{Float64}

Build the observation target vector from cached WOA-on-ORCA fields, sliced
to the same tropical / upper-ocean region as the model output.
"""
function process_woa_target(woa_file::AbstractString;
                            lat_range = DEFAULT_LAT_RANGE,
                            z_min     = DEFAULT_Z_MIN)
    T, S = load_woa_on_orca(woa_file)
    T_vec = extract_tropics_top(T; lat_range, z_min)
    S_vec = extract_tropics_top(S; lat_range, z_min)
    return vcat(T_vec, S_vec)
end

"""
    build_diagonal_covariance(Y_target; T_variance, S_variance) -> Diagonal

Diagonal observation covariance for a flat `[T..., S...]` vector with equal
T and S sections.
"""
function build_diagonal_covariance(Y_target::AbstractVector;
                                   T_variance::Real = 0.5^2,
                                   S_variance::Real = (0.5/4)^2)
    n_total = length(Y_target)
    iseven(n_total) || error("Y_target must have even length (T and S concatenated); got $n_total")
    n_field = n_total ÷ 2
    diag_values = vcat(fill(T_variance, n_field), fill(S_variance, n_field))
    return Diagonal(diag_values)
end
