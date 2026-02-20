# unflatten_utilities.jl
# Utilities for unflattening observation/model vectors back to spatially meaningful T and S fields
#
# The flattened observation vectors have the structure:
#   [month1_T, month1_S, month2_T, month2_S, ..., month12_T, month12_S]
# where T and S are flattened arrays with NaN values removed.

using Oceananigans
using Oceananigans.Grids: znodes, φnodes, λnodes
using Oceananigans.Fields: location, Field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: on_architecture
using JLD2
using NaNStatistics

# ============================================
# Helper Functions
# ============================================

"""
    compute_dz_weights(grid, z_indices)

Compute vertical weights based on grid cell thickness (Δz).
This ensures all depth levels contribute proportionally to their thickness,
compensating for vertical grid stretching.

Returns normalized weights that sum to 1.
"""
function compute_dz_weights(grid, z_indices)
    # Get z faces
    zf = znodes(grid, Face())

    # Compute Δz for each cell in z_indices
    Δz = [zf[k+1] - zf[k] for k in z_indices]

    # Normalize so weights sum to 1
    Δz_normalized = Δz ./ sum(Δz)

    return Δz_normalized
end

# ============================================
# UnflattenMap Structure
# ============================================

"""
    UnflattenMap

Structure that stores all information needed to unflatten a flattened observation/model vector
back to spatially meaningful T and S fields.

Fields:
- `grid`: The target grid used for regridding
- `lat_indices`: Indices of the latitude dimension used (-52 to 52)
- `z_indices`: Indices of the z dimension used (z >= z_min)
- `T_mask`: Boolean mask for T field (true = valid data, false = NaN/land)
- `S_mask`: Boolean mask for S field (true = valid data, false = NaN/land)
- `section_shape`: Shape of the extracted section (Nx, Ny, Nz)
- `n_T`: Number of valid T values per month
- `n_S`: Number of valid S values per month
- `zonal_average`: Whether zonal averaging was applied
- `apply_dz_weighting`: Whether dz weighting was applied
- `z_weights`: The dz weights used (for unweighting if needed)
"""
struct UnflattenMap
    grid::Any
    lat_indices::UnitRange{Int}
    z_indices::Vector{Int}
    T_mask::BitArray
    S_mask::BitArray
    section_shape::Tuple{Int, Int, Int}
    n_T::Int
    n_S::Int
    zonal_average::Bool
    apply_dz_weighting::Bool
    z_weights::Vector{Float64}
end

# ============================================
# Building UnflattenMap
# ============================================

"""
    build_unflatten_map(obs_path, zonal_average; apply_dz_weighting=false, z_min=-1000)

Build an UnflattenMap from observation data that can be used to unflatten vectors
back to spatial T and S fields.

Arguments:
- `obs_path`: Path to observation data (directory containing T.jld2 and S.jld2)
- `zonal_average`: Whether zonal averaging is used
- `apply_dz_weighting`: Whether dz weighting is applied
- `z_min`: Minimum depth to include (default -1000m)

Returns an UnflattenMap structure.
"""
function build_unflatten_map(obs_path, zonal_average; apply_dz_weighting=false, z_min=-1000)
    T_filepath = joinpath(obs_path, "T.jld2")

    T_afts = jldopen(T_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    T_data = T_afts.data
    T_data = on_architecture(CPU(), T_data)

    LX, LY, LZ = location(T_data)
    grid = T_data.grid

    φᶜ = φnodes(grid, LX(), LY(), LZ())
    zᶜ = znodes(grid, LX(), LY(), LZ())

    # Get latitude and z indices (same as in extract_field_section)
    latitude_range = (-52, 52)
    φmin, φmax = latitude_range
    lat_indices = findfirst(x -> x >= φmin, φᶜ):findlast(x -> x <= φmax, φᶜ)
    z_indices = findall(z -> z >= z_min, zᶜ)

    # Compute dz weights using compute_dz_weights from including scope
    if apply_dz_weighting
        z_weights = compute_dz_weights(grid, z_indices)
    else
        z_weights = ones(length(z_indices))
    end

    # Get the section shape
    Nx = size(T_data, 1)
    Ny = length(lat_indices)
    Nz = length(z_indices)

    # Build mask by processing one month's data
    mask_immersed_field!(T_data[1], NaN)
    T_section = interior(T_data[1], :, lat_indices, z_indices)

    if zonal_average
        T_section = nanmean(T_section, dims=1)
        section_shape = (1, Ny, Nz)
    else
        section_shape = (Nx, Ny, Nz)
    end

    T_mask = .!isnan.(T_section)
    n_T = sum(T_mask)

    # For S, assume same mask structure as T (same grid, same land mask)
    S_mask = copy(T_mask)
    n_S = sum(S_mask)

    return UnflattenMap(grid, lat_indices, z_indices, T_mask, S_mask,
                        section_shape, n_T, n_S, zonal_average, apply_dz_weighting, z_weights)
end

"""
    build_unflatten_map_from_grid(grid_jld2_path, zonal_average; apply_dz_weighting=false, z_min=-1000,
                                   sample_field=nothing)

Build an UnflattenMap from a saved grid file (e.g., 4deg_grids_and_regridder.jld2).
This is useful when you don't have access to observation files but have the grid.

Arguments:
- `grid_jld2_path`: Path to JLD2 file containing "target_grid"
- `zonal_average`: Whether zonal averaging is used
- `apply_dz_weighting`: Whether dz weighting is applied
- `z_min`: Minimum depth to include (default -1000m)
- `sample_field`: Optional field to build mask from

Returns an UnflattenMap structure (T_mask and S_mask will need to be set from actual data).
"""
function build_unflatten_map_from_grid(grid_jld2_path, zonal_average;
                                        apply_dz_weighting=false, z_min=-1000,
                                        sample_field=nothing)
    target_grid = jldopen(grid_jld2_path, "r") do file
        return file["target_grid"]
    end

    # Use Center location for T and S fields
    φᶜ = φnodes(target_grid, Center(), Center(), Center())
    zᶜ = znodes(target_grid, Center(), Center(), Center())

    # Get latitude and z indices
    latitude_range = (-52, 52)
    φmin, φmax = latitude_range
    lat_indices = findfirst(x -> x >= φmin, φᶜ):findlast(x -> x <= φmax, φᶜ)
    z_indices = findall(z -> z >= z_min, zᶜ)

    # Compute dz weights
    if apply_dz_weighting
        z_weights = compute_dz_weights(target_grid, z_indices)
    else
        z_weights = ones(length(z_indices))
    end

    # Get the section shape
    Nx = size(target_grid, 1)
    Ny = length(lat_indices)
    Nz = length(z_indices)

    if zonal_average
        section_shape = (1, Ny, Nz)
    else
        section_shape = (Nx, Ny, Nz)
    end

    # If a sample field is provided, build the mask from it
    if !isnothing(sample_field)
        sample_field = on_architecture(CPU(), sample_field)
        mask_immersed_field!(sample_field, NaN)
        T_section = interior(sample_field, :, lat_indices, z_indices)

        if zonal_average
            T_section = nanmean(T_section, dims=1)
        end

        T_mask = .!isnan.(T_section)
        S_mask = copy(T_mask)
    else
        # Without a sample field, assume all values are valid (will be overwritten)
        T_mask = trues(section_shape)
        S_mask = trues(section_shape)
    end

    n_T = sum(T_mask)
    n_S = sum(S_mask)

    return UnflattenMap(target_grid, lat_indices, z_indices, T_mask, S_mask,
                        section_shape, n_T, n_S, zonal_average, apply_dz_weighting, z_weights)
end

# ============================================
# Unflattening Functions
# ============================================

"""
    unflatten_monthly_vector(vec, unflatten_map)

Unflatten a single month's observation/model vector back to T and S fields.

Arguments:
- `vec`: Flattened vector of length n_T + n_S
- `unflatten_map`: UnflattenMap structure

Returns:
- `T_field`: 3D array of temperature (with NaN for land/bathymetry)
- `S_field`: 3D array of salinity (with NaN for land/bathymetry)
"""
function unflatten_monthly_vector(vec, unflatten_map::UnflattenMap)
    (; T_mask, S_mask, section_shape, n_T, n_S, apply_dz_weighting, z_weights) = unflatten_map

    expected_length = n_T + n_S
    if length(vec) != expected_length
        error("Vector length $(length(vec)) does not match expected $expected_length (n_T=$n_T + n_S=$n_S)")
    end

    # Split into T and S parts
    T_vec = vec[1:n_T]
    S_vec = vec[n_T+1:end]

    # Initialize fields with NaN
    T_field = fill(NaN, section_shape)
    S_field = fill(NaN, section_shape)

    # Undo dz weighting if it was applied
    if apply_dz_weighting
        # z_weights were applied as: weighted = z_weights .* unweighted
        # To undo: unweighted = weighted ./ z_weights
        # The z_weights have shape (1, 1, Nz), so we need to match indices
        z_weight_array = reshape(z_weights, 1, 1, :)
    end

    # Fill in valid values
    T_field[T_mask] .= T_vec
    S_field[S_mask] .= S_vec

    # Undo dz weighting
    if apply_dz_weighting
        T_field ./= z_weight_array
        S_field ./= z_weight_array
    end

    return T_field, S_field
end

"""
    unflatten_yearly_vector(vec, unflatten_map; n_months=12)

Unflatten a full year's observation/model vector back to monthly T and S fields.

Arguments:
- `vec`: Flattened vector of length n_months * (n_T + n_S)
- `unflatten_map`: UnflattenMap structure
- `n_months`: Number of months (default 12)

Returns:
- `T_fields`: 4D array of temperature (Nx, Ny, Nz, n_months)
- `S_fields`: 4D array of salinity (Nx, Ny, Nz, n_months)
"""
function unflatten_yearly_vector(vec, unflatten_map::UnflattenMap; n_months=12)
    (; section_shape, n_T, n_S) = unflatten_map

    monthly_length = n_T + n_S
    expected_length = n_months * monthly_length

    if length(vec) != expected_length
        error("Vector length $(length(vec)) does not match expected $expected_length ($n_months months × $monthly_length per month)")
    end

    # Initialize 4D arrays
    T_fields = fill(NaN, section_shape..., n_months)
    S_fields = fill(NaN, section_shape..., n_months)

    # Unflatten each month
    for m in 1:n_months
        start_idx = (m - 1) * monthly_length + 1
        end_idx = m * monthly_length
        monthly_vec = vec[start_idx:end_idx]

        T_field, S_field = unflatten_monthly_vector(monthly_vec, unflatten_map)
        T_fields[:, :, :, m] .= T_field
        S_fields[:, :, :, m] .= S_field
    end

    return T_fields, S_fields
end

# ============================================
# Coordinate and Helper Functions
# ============================================

"""
    get_spatial_coordinates(unflatten_map)

Get the latitude and depth coordinates corresponding to the unflattened fields.

Returns:
- `lats`: Latitude coordinates for the y dimension
- `depths`: Depth coordinates for the z dimension
- `lons`: Longitude coordinates for the x dimension (if not zonally averaged)
"""
function get_spatial_coordinates(unflatten_map::UnflattenMap)
    (; grid, lat_indices, z_indices, zonal_average) = unflatten_map

    φᶜ = φnodes(grid, Center(), Center(), Center())
    zᶜ = znodes(grid, Center(), Center(), Center())
    λᶜ = λnodes(grid, Center(), Center(), Center())

    lats = φᶜ[lat_indices]
    depths = zᶜ[z_indices]

    if zonal_average
        lons = nothing
    else
        lons = λᶜ
    end

    return (; lons, lats, depths)
end

"""
    unflatten_to_named_tuple(vec, unflatten_map; n_months=12)

Convenience function that returns unflattened data as a named tuple with coordinates.

Returns a NamedTuple with fields:
- `T`: Temperature array (3D for single month, 4D for yearly)
- `S`: Salinity array (3D for single month, 4D for yearly)
- `lons`: Longitude coordinates (nothing if zonally averaged)
- `lats`: Latitude coordinates
- `depths`: Depth coordinates
- `months`: Month indices (1:12 for yearly data)
"""
function unflatten_to_named_tuple(vec, unflatten_map::UnflattenMap; n_months=12)
    (; n_T, n_S) = unflatten_map
    monthly_length = n_T + n_S

    coords = get_spatial_coordinates(unflatten_map)

    if length(vec) == monthly_length
        # Single month
        T, S = unflatten_monthly_vector(vec, unflatten_map)
        return (; T, S, coords..., months=nothing)
    else
        # Multiple months
        T, S = unflatten_yearly_vector(vec, unflatten_map; n_months)
        return (; T, S, coords..., months=1:n_months)
    end
end

"""
    compute_monthly_lengths(unflatten_map)

Return the lengths of the T and S vectors per month, useful for splitting vectors.
"""
function compute_monthly_lengths(unflatten_map::UnflattenMap)
    return (; n_T=unflatten_map.n_T, n_S=unflatten_map.n_S,
              monthly_total=unflatten_map.n_T + unflatten_map.n_S)
end
