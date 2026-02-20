# unflatten_covariance.jl
# Utilities for unflattening the observation covariance matrix to spatially meaningful fields
#
# The covariance matrix is built from flattened observation vectors with structure:
#   [month1_T, month1_S, month2_T, month2_S, ..., month12_T, month12_S]
#
# This script provides functions to:
# 1. Extract diagonal (variances) and unflatten to spatial T/S variance fields
# 2. Extract covariance between specific spatial points
# 3. Compute derived statistics (correlation, variance ratios, etc.)

using LinearAlgebra
using JLD2
using Statistics
using EnsembleKalmanProcesses: SVDplusD

# Include unflatten utilities for UnflattenMap
include(joinpath(@__DIR__, "unflatten_utilities.jl"))

# Include data processing for build_observation_covariance
include(joinpath(@__DIR__, "data_processing.jl"))

# ============================================
# Covariance Index Mapping
# ============================================

"""
    CovarianceIndexMap

Structure for mapping between flattened covariance indices and spatial/temporal coordinates.

Fields:
- `unflatten_map`: The UnflattenMap for spatial structure
- `n_months`: Number of months in the yearly vector
- `monthly_length`: Length of one month's flattened vector (n_T + n_S)
- `yearly_length`: Total length of yearly vector (n_months * monthly_length)
"""
struct CovarianceIndexMap
    unflatten_map::UnflattenMap
    n_months::Int
    monthly_length::Int
    yearly_length::Int
end

"""
    build_covariance_index_map(unflatten_map; n_months=12)

Build a CovarianceIndexMap from an UnflattenMap.
"""
function build_covariance_index_map(unflatten_map::UnflattenMap; n_months=12)
    monthly_length = unflatten_map.n_T + unflatten_map.n_S
    yearly_length = n_months * monthly_length
    return CovarianceIndexMap(unflatten_map, n_months, monthly_length, yearly_length)
end

"""
    flat_index_to_components(idx, cov_map)

Convert a flat index into its components: (month, variable, spatial_index).

Arguments:
- `idx`: Flat index (1-based)
- `cov_map`: CovarianceIndexMap

Returns:
- `month`: Month index (1-12)
- `variable`: :T or :S
- `spatial_idx`: Index within the valid T or S values for that month
"""
function flat_index_to_components(idx::Int, cov_map::CovarianceIndexMap)
    (; unflatten_map, n_months, monthly_length) = cov_map
    (; n_T, n_S) = unflatten_map

    # Determine which month
    month = div(idx - 1, monthly_length) + 1

    # Index within the month
    idx_in_month = mod1(idx, monthly_length)

    # Determine if T or S
    if idx_in_month <= n_T
        variable = :T
        spatial_idx = idx_in_month
    else
        variable = :S
        spatial_idx = idx_in_month - n_T
    end

    return (; month, variable, spatial_idx)
end

"""
    components_to_flat_index(month, variable, spatial_idx, cov_map)

Convert component indices to a flat index.

Arguments:
- `month`: Month index (1-12)
- `variable`: :T or :S
- `spatial_idx`: Index within the valid T or S values
- `cov_map`: CovarianceIndexMap

Returns:
- Flat index (1-based)
"""
function components_to_flat_index(month::Int, variable::Symbol, spatial_idx::Int, cov_map::CovarianceIndexMap)
    (; unflatten_map, monthly_length) = cov_map
    (; n_T) = unflatten_map

    base_idx = (month - 1) * monthly_length

    if variable == :T
        return base_idx + spatial_idx
    else  # :S
        return base_idx + n_T + spatial_idx
    end
end

"""
    spatial_idx_to_cartesian(spatial_idx, mask)

Convert a spatial index (within valid values) to Cartesian indices in the section.

Arguments:
- `spatial_idx`: Index within the valid (non-NaN) values
- `mask`: Boolean mask array (true = valid)

Returns:
- Cartesian index (i, j, k) in the section
"""
function spatial_idx_to_cartesian(spatial_idx::Int, mask::BitArray)
    valid_indices = findall(mask)
    return Tuple(valid_indices[spatial_idx])
end

"""
    cartesian_to_spatial_idx(i, j, k, mask)

Convert Cartesian indices to a spatial index (within valid values).

Returns nothing if the point is masked (invalid).
"""
function cartesian_to_spatial_idx(i::Int, j::Int, k::Int, mask::BitArray)
    if !mask[i, j, k]
        return nothing
    end

    valid_indices = findall(mask)
    target = CartesianIndex(i, j, k)

    for (idx, ci) in enumerate(valid_indices)
        if ci == target
            return idx
        end
    end

    return nothing
end

# ============================================
# Variance Extraction and Unflattening
# ============================================

"""
    extract_variances(covariance)

Extract the diagonal elements (variances) from a covariance matrix.

Handles both standard matrices and SVDplusD covariance structures.
"""
function extract_variances(covariance::AbstractMatrix)
    return diag(covariance)
end

function extract_variances(covariance::SVDplusD)
    # SVDplusD stores: U * S * U' + D
    # Diagonal is: sum of (U_i .* S .* U_i) + D_ii
    # For variance extraction, we compute it directly
    svd_part = covariance.svd_cov
    d_part = covariance.diag_cov

    # U * S * U' diagonal elements
    n = size(svd_part.U, 1)
    variances = zeros(n)

    for i in 1:n
        # Diagonal of U * Diagonal(S) * U' at (i,i) is sum_j (U[i,j]^2 * S[j])
        variances[i] = sum(svd_part.U[i, j]^2 * svd_part.S[j] for j in 1:length(svd_part.S))
    end

    # Add diagonal part
    variances .+= diag(d_part)

    return variances
end

"""
    unflatten_variances(variances, cov_map)

Unflatten a variance vector to monthly T and S variance fields.

Arguments:
- `variances`: Vector of variances (length = yearly_length)
- `cov_map`: CovarianceIndexMap

Returns:
- `T_var`: Temperature variance (Nx, Ny, Nz, n_months)
- `S_var`: Salinity variance (Nx, Ny, Nz, n_months)
"""
function unflatten_variances(variances::Vector, cov_map::CovarianceIndexMap)
    (; unflatten_map, n_months, monthly_length) = cov_map
    (; T_mask, S_mask, section_shape, n_T, n_S) = unflatten_map

    if length(variances) != n_months * monthly_length
        error("Variance vector length $(length(variances)) doesn't match expected $(n_months * monthly_length)")
    end

    # Initialize output arrays
    T_var = fill(NaN, section_shape..., n_months)
    S_var = fill(NaN, section_shape..., n_months)

    for m in 1:n_months
        # Get variances for this month
        start_idx = (m - 1) * monthly_length + 1
        T_var_month = variances[start_idx:start_idx + n_T - 1]
        S_var_month = variances[start_idx + n_T:start_idx + monthly_length - 1]

        # Fill in the spatial fields
        T_var_field = fill(NaN, section_shape)
        S_var_field = fill(NaN, section_shape)

        T_var_field[T_mask] .= T_var_month
        S_var_field[S_mask] .= S_var_month

        T_var[:, :, :, m] .= T_var_field
        S_var[:, :, :, m] .= S_var_field
    end

    return T_var, S_var
end

"""
    unflatten_covariance_diagonal(covariance, unflatten_map; n_months=12)

Convenience function to extract and unflatten the covariance diagonal (variances).

Returns:
- `T_var`: Temperature variance (Nx, Ny, Nz, n_months)
- `S_var`: Salinity variance (Nx, Ny, Nz, n_months)
- `cov_map`: The CovarianceIndexMap for further analysis
"""
function unflatten_covariance_diagonal(covariance, unflatten_map::UnflattenMap; n_months=12)
    cov_map = build_covariance_index_map(unflatten_map; n_months)
    variances = extract_variances(covariance)
    T_var, S_var = unflatten_variances(variances, cov_map)
    return T_var, S_var, cov_map
end

# ============================================
# Covariance/Correlation Extraction
# ============================================

"""
    get_covariance_element(covariance, idx1, idx2)

Get a single element from the covariance matrix.
"""
function get_covariance_element(covariance::AbstractMatrix, idx1::Int, idx2::Int)
    return covariance[idx1, idx2]
end

function get_covariance_element(covariance::SVDplusD, idx1::Int, idx2::Int)
    svd_part = covariance.svd_cov
    d_part = covariance.diag_cov

    # SVD part: U * S * U' at (idx1, idx2)
    svd_element = sum(svd_part.U[idx1, j] * svd_part.S[j] * svd_part.U[idx2, j]
                      for j in 1:length(svd_part.S))

    # Diagonal part contributes only if idx1 == idx2
    if idx1 == idx2
        return svd_element + d_part[idx1, idx1]
    else
        return svd_element
    end
end

"""
    extract_covariance_row(covariance, row_idx)

Extract a single row from the covariance matrix.
"""
function extract_covariance_row(covariance::AbstractMatrix, row_idx::Int)
    return covariance[row_idx, :]
end

function extract_covariance_row(covariance::SVDplusD, row_idx::Int)
    svd_part = covariance.svd_cov
    d_part = covariance.diag_cov

    n = size(svd_part.U, 1)
    row = zeros(n)

    # SVD part: U[row_idx, :] * S * U'
    for j in 1:n
        row[j] = sum(svd_part.U[row_idx, k] * svd_part.S[k] * svd_part.U[j, k]
                     for k in 1:length(svd_part.S))
    end

    # Add diagonal element
    row[row_idx] += d_part[row_idx, row_idx]

    return row
end

"""
    get_point_covariance(covariance, month1, var1, i1, j1, k1,
                         month2, var2, i2, j2, k2, cov_map)

Get the covariance between two specific spatial/temporal points.

Arguments:
- `covariance`: The covariance matrix or SVDplusD object
- `month1`, `month2`: Month indices (1-12)
- `var1`, `var2`: Variables (:T or :S)
- `i1, j1, k1`, `i2, j2, k2`: Cartesian indices in the section
- `cov_map`: CovarianceIndexMap

Returns the covariance value, or NaN if either point is masked.
"""
function get_point_covariance(covariance,
                              month1::Int, var1::Symbol, i1::Int, j1::Int, k1::Int,
                              month2::Int, var2::Symbol, i2::Int, j2::Int, k2::Int,
                              cov_map::CovarianceIndexMap)
    (; unflatten_map) = cov_map
    mask1 = var1 == :T ? unflatten_map.T_mask : unflatten_map.S_mask
    mask2 = var2 == :T ? unflatten_map.T_mask : unflatten_map.S_mask

    spatial_idx1 = cartesian_to_spatial_idx(i1, j1, k1, mask1)
    spatial_idx2 = cartesian_to_spatial_idx(i2, j2, k2, mask2)

    if isnothing(spatial_idx1) || isnothing(spatial_idx2)
        return NaN
    end

    flat_idx1 = components_to_flat_index(month1, var1, spatial_idx1, cov_map)
    flat_idx2 = components_to_flat_index(month2, var2, spatial_idx2, cov_map)

    return get_covariance_element(covariance, flat_idx1, flat_idx2)
end

"""
    extract_covariance_map_for_point(covariance, month, variable, i, j, k, cov_map)

Extract the covariance of one point with all other points, unflattened to spatial fields.

This shows how one spatial/temporal point covaries with the entire domain.

Arguments:
- `covariance`: The covariance matrix or SVDplusD object
- `month`: Month index (1-12) of the reference point
- `variable`: Variable (:T or :S) of the reference point
- `i, j, k`: Cartesian indices of the reference point
- `cov_map`: CovarianceIndexMap

Returns:
- `T_cov`: Covariance with T field (Nx, Ny, Nz, n_months)
- `S_cov`: Covariance with S field (Nx, Ny, Nz, n_months)
"""
function extract_covariance_map_for_point(covariance,
                                           month::Int, variable::Symbol,
                                           i::Int, j::Int, k::Int,
                                           cov_map::CovarianceIndexMap)
    (; unflatten_map, n_months, monthly_length) = cov_map
    mask = variable == :T ? unflatten_map.T_mask : unflatten_map.S_mask

    spatial_idx = cartesian_to_spatial_idx(i, j, k, mask)
    if isnothing(spatial_idx)
        error("Point ($i, $j, $k) is masked for variable $variable")
    end

    flat_idx = components_to_flat_index(month, variable, spatial_idx, cov_map)

    # Extract the row
    cov_row = extract_covariance_row(covariance, flat_idx)

    # Unflatten to T and S covariance fields
    # Use the same structure as variances
    T_cov, S_cov = unflatten_variances(cov_row, cov_map)

    return T_cov, S_cov
end

"""
    compute_correlation_from_covariance(cov_val, var1, var2)

Compute correlation from covariance: ρ = cov / sqrt(var1 * var2)
"""
function compute_correlation_from_covariance(cov_val, var1, var2)
    if var1 <= 0 || var2 <= 0
        return NaN
    end
    return cov_val / sqrt(var1 * var2)
end

"""
    extract_correlation_map_for_point(covariance, month, variable, i, j, k, cov_map)

Extract the correlation of one point with all other points.

Similar to extract_covariance_map_for_point but normalized to correlation coefficients.
"""
function extract_correlation_map_for_point(covariance,
                                            month::Int, variable::Symbol,
                                            i::Int, j::Int, k::Int,
                                            cov_map::CovarianceIndexMap)
    # Get covariance map
    T_cov, S_cov = extract_covariance_map_for_point(covariance, month, variable, i, j, k, cov_map)

    # Get variances
    variances = extract_variances(covariance)
    T_var, S_var = unflatten_variances(variances, cov_map)

    # Get reference point variance
    (; unflatten_map) = cov_map
    mask = variable == :T ? unflatten_map.T_mask : unflatten_map.S_mask
    ref_var = variable == :T ? T_var[i, j, k, month] : S_var[i, j, k, month]

    # Compute correlations
    T_corr = T_cov ./ sqrt.(ref_var .* T_var)
    S_corr = S_cov ./ sqrt.(ref_var .* S_var)

    return T_corr, S_corr
end

# ============================================
# Summary Statistics
# ============================================

"""
    compute_spatial_variance_summary(T_var, S_var, coords; depth_bins=nothing)

Compute summary statistics of variances by depth or region.

Arguments:
- `T_var`: Temperature variance array (Nx, Ny, Nz, n_months)
- `S_var`: Salinity variance array (Nx, Ny, Nz, n_months)
- `coords`: Coordinates from get_spatial_coordinates
- `depth_bins`: Optional depth bins for averaging (e.g., [0, -100, -500, -1000])

Returns a named tuple with summary statistics.
"""
function compute_spatial_variance_summary(T_var, S_var, coords; depth_bins=nothing)
    depths = coords.depths

    # Mean variance across months and horizontal dimensions
    T_var_mean = dropdims(nanmean(T_var, dims=(1, 2, 4)), dims=(1, 2, 4))
    S_var_mean = dropdims(nanmean(S_var, dims=(1, 2, 4)), dims=(1, 2, 4))

    # Mean variance by month (averaged over space)
    T_var_monthly = [nanmean(T_var[:, :, :, m]) for m in 1:size(T_var, 4)]
    S_var_monthly = [nanmean(S_var[:, :, :, m]) for m in 1:size(S_var, 4)]

    summary = (;
        depths,
        T_var_by_depth = T_var_mean,
        S_var_by_depth = S_var_mean,
        T_var_by_month = T_var_monthly,
        S_var_by_month = S_var_monthly,
        T_var_total = nanmean(T_var),
        S_var_total = nanmean(S_var),
    )

    return summary
end

# ============================================
# Convenience Function
# ============================================

"""
    build_variance_fields(obs_paths, zonal_average; model_error_frac=0.0, error_regularizer=1e-4)

Build T and S variance fields from observation data.

This is a convenience function that:
1. Builds the observation covariance from the provided paths
2. Builds an unflatten map from the first observation path
3. Extracts and unflattens the covariance diagonal to variance fields
4. Returns all results as a named tuple

Arguments:
- `obs_paths`: Vector of paths to observation data directories
- `zonal_average`: Whether to use zonal averaging
- `model_error_frac`: Fraction of mean field values for model error covariance (default 0.0)
- `error_regularizer`: Regularization term for observation covariance (default 1e-4)

Returns a NamedTuple with fields:
- `T_var`: Temperature variance (Nx, Ny, Nz, n_months)
- `S_var`: Salinity variance (Nx, Ny, Nz, n_months)
- `coords`: Spatial coordinates (lons, lats, depths)
- `cov_map`: CovarianceIndexMap for further analysis
- `unflatten_map`: UnflattenMap for further analysis
- `covariance`: The full covariance matrix/object
- `summary`: Summary statistics
"""
function build_variance_fields(obs_paths, zonal_average; model_error_frac=0.0, error_regularizer=1e-4)
    @info "Building observation covariance from $(length(obs_paths)) years..."
    covariance, Y_all = build_observation_covariance(obs_paths, zonal_average; model_error_frac, error_regularizer)

    @info "Building unflatten map..."
    unflatten_map = build_unflatten_map(first(obs_paths), zonal_average; apply_dz_weighting=false)

    @info "Extracting and unflattening variances..."
    T_var, S_var, cov_map = unflatten_covariance_diagonal(covariance, unflatten_map)

    coords = get_spatial_coordinates(unflatten_map)
    summary = compute_spatial_variance_summary(T_var, S_var, coords)

    @info "Variance fields shape: $(size(T_var))"
    @info "  Temperature variance range: $(nanminimum(T_var)) to $(nanmaximum(T_var))"
    @info "  Salinity variance range: $(nanminimum(S_var)) to $(nanmaximum(S_var))"

    return (;
        T_var,
        S_var,
        coords,
        cov_map,
        unflatten_map,
        covariance,
        summary
    )
end
