# data_processing.jl
# Data processing utilities for CATKE calibration
# Uses 4-degree grid for regridding model output to match observations
#
# For CATKE calibration:
# - Forward model runs 2 years, outputs monthly averages for year 2 (to avoid initialization shock)
# - Observations are monthly-averaged ECCO4 data (12 months per year)
# - Comparison uses upper ocean only (z >= -1000m) where CATKE affects vertical mixing

using Oceananigans
using Oceananigans.Grids: znodes, φnodes
using Oceananigans.Fields: location, Field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: on_architecture
using ClimaOcean
using ClimaOcean.DataWrangling
using XESMF
using JLD2
using NaNStatistics
using Glob
using Statistics
using LinearAlgebra
using EnsembleKalmanProcesses: tsvd_cov_from_samples, SVDplusD
using ClimaOceanCalibration

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

"""
    regrid_model_data(simdir, target_grid, regridder, month_name)

Regrid model output from 0.5° tripolar grid to 4° lat-lon grid.

Arguments:
- `simdir`: Directory containing model output files
- `target_grid`: Target grid for regridding (4° lat-lon)
- `regridder`: XESMF regridder object
- `month_name`: Month symbol (e.g., :jan, :feb) to specify which monthly average file to read
- `buoyancy`: If true, process buoyancy as well as temperature and salinity
"""
function regrid_model_data(simdir, target_grid, regridder, month_name; buoyancy=false)
    filepath = joinpath(simdir, "ocean_$(month_name)_average.jld2")

    T_data = FieldTimeSeries(filepath, "T", backend=OnDisk())
    S_data = FieldTimeSeries(filepath, "S", backend=OnDisk())

    T_target = CenterField(target_grid)
    S_target = CenterField(target_grid)

    # Each monthly file should have 2 time indices:
    # - Time index 1: initial snapshot at t=0
    # - Time index 2: the monthly average
    # Warn if there are more than 2, and always use the 2nd timestep.
    Nt = length(T_data.times)
    if Nt > 2
        @warn "Expected 2 time indices in $filepath, found $Nt. Using 2nd time index."
    end
    regrid!(T_target, regridder, T_data[2])
    regrid!(S_target, regridder, S_data[2])

    if buoyancy
        b_data = FieldTimeSeries(filepath, "b", backend=OnDisk())
        b_target = CenterField(target_grid)
        regrid!(b_target, regridder, b_data[2])
        return T_target, S_target, b_target
    end
    
    return T_target, S_target
end

"""
    extract_field_section(fts::FieldTimeSeries, latitude_range;
                          apply_dz_weighting=false, time_index=nothing,
                          z_min=-1000)

Extract a latitude section from a FieldTimeSeries with optional dz-based weighting.
If time_index is not provided, uses the last time index.

When apply_dz_weighting=true, applies weights proportional to grid cell thickness (Δz).
This compensates for vertical grid stretching in the loss function.

For CATKE calibration, we only use upper ocean data (z >= z_min, default -1000m)
since CATKE primarily affects vertical mixing in the mixed layer and upper ocean.
"""
function extract_field_section(fts::FieldTimeSeries, latitude_range;
                               apply_dz_weighting=false, time_index=nothing,
                               z_min=-1000)
    fts = on_architecture(CPU(), fts)
    LX, LY, LZ = location(fts)
    grid = fts.grid

    φᶜ = φnodes(grid, LX(), LY(), LZ())
    zᶜ = znodes(grid, LX(), LY(), LZ())

    φmin, φmax = latitude_range

    lat_indices = findfirst(x -> x >= φmin, φᶜ):findlast(x -> x <= φmax, φᶜ)

    # Filter by depth: only use z >= z_min (upper ocean)
    z_indices = findall(z -> z >= z_min, zᶜ)

    # Compute weights based on grid cell thickness if requested
    if apply_dz_weighting
        z_weights = compute_dz_weights(grid, z_indices)
    else
        z_weights = ones(length(z_indices))
    end

    Nt = length(fts.times)
    for t in 1:Nt
        mask_immersed_field!(fts[t], NaN)
    end

    t_idx = isnothing(time_index) ? Nt : time_index
    field_section = reshape(z_weights, 1, 1, :) .* interior(fts[t_idx], :, lat_indices, z_indices)

    return field_section
end

"""
    extract_field_section(field::Field, latitude_range;
                          apply_dz_weighting=false, z_min=-1000)

Extract a latitude section from a Field with optional dz-based weighting.

When apply_dz_weighting=true, applies weights proportional to grid cell thickness (Δz).
This compensates for vertical grid stretching in the loss function.

For CATKE calibration, we only use upper ocean data (z >= z_min, default -1000m)
since CATKE primarily affects vertical mixing in the mixed layer and upper ocean.
"""
function extract_field_section(field::Field, latitude_range;
                               apply_dz_weighting=false, z_min=-1000)
    field = on_architecture(CPU(), field)
    LX, LY, LZ = location(field)
    grid = field.grid

    φᶜ = φnodes(grid, LX(), LY(), LZ())
    zᶜ = znodes(grid, LX(), LY(), LZ())

    φmin, φmax = latitude_range

    lat_indices = findfirst(x -> x >= φmin, φᶜ):findlast(x -> x <= φmax, φᶜ)

    # Filter by depth: only use z >= z_min (upper ocean)
    z_indices = findall(z -> z >= z_min, zᶜ)

    # Compute weights based on grid cell thickness if requested
    if apply_dz_weighting
        z_weights = compute_dz_weights(grid, z_indices)
    else
        z_weights = ones(length(z_indices))
    end

    mask_immersed_field!(field, NaN)

    field_section = reshape(z_weights, 1, 1, :) .* interior(field, :, lat_indices, z_indices)

    return field_section
end

extract_midlatitude_section(fts; kwargs...) = extract_field_section(fts, (-52, 52); kwargs...)

"""
    process_observation(obs_path, zonal_average; month_index=nothing, apply_dz_weighting=false)

Process observation data from a given path.

For CATKE calibration with monthly averages:
- obs_path should contain T.jld2 and S.jld2 files with 12 monthly time indices
- month_index (1-12) selects which month to use; if nothing, uses last time index
- apply_dz_weighting: if true, applies weights proportional to grid cell thickness
"""
function process_observation(obs_path, zonal_average; month_index=nothing, apply_dz_weighting=false)
    T_filepath = joinpath(obs_path, "T.jld2")
    S_filepath = joinpath(obs_path, "S.jld2")

    T_afts = jldopen(T_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    S_afts = jldopen(S_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    T_data = T_afts.data
    S_data = S_afts.data

    # Extract mid-latitude ocean section with specified month
    T_section = extract_midlatitude_section(T_data; time_index=month_index, apply_dz_weighting)
    S_section = extract_midlatitude_section(S_data; time_index=month_index, apply_dz_weighting)

    if zonal_average
        T_section = nanmean(T_section, dims=1)
        S_section = nanmean(S_section, dims=1)
    end

    return vcat(T_section[.!isnan.(T_section)], S_section[.!isnan.(S_section)])
end

"""
    process_monthly_observations(obs_path, zonal_average; apply_dz_weighting=false)

Process all 12 monthly observations from a given path and return as a matrix.
Each column corresponds to one month (Jan-Dec).
"""
function process_monthly_observations(obs_path, zonal_average; apply_dz_weighting=false)
    monthly_obs = [process_observation(obs_path, zonal_average; month_index=m, apply_dz_weighting) for m in 1:12]
    return hcat(monthly_obs...)
end

"""
    process_member_data(simdir, zonal_average; apply_dz_weighting=false)

Process model output from a single ensemble member.

For CATKE calibration, this reads the 12 monthly average output files and
concatenates them into a single observation vector matching the format of
the target observations.
"""
function process_member_data(simdir, zonal_average; apply_dz_weighting=false)
    month_names = [:jan, :feb, :mar, :apr, :may, :jun, :jul, :aug, :sep, :oct, :nov, :dec]

    # Load target grid and regridder once
    target_grid, regridder = jldopen(joinpath(pwd(), "examples", "CATKE_calibration", "4deg_grids_and_regridder.jld2"), "r") do file
        return file["target_grid"], file["regridder"]
    end

    monthly_outputs = []
    for month_name in month_names
        filepath = joinpath(simdir, "ocean_$(month_name)_average.jld2")
        if !isfile(filepath)
            @warn "Missing monthly file: $filepath"
            continue
        end

        T_target, S_target = regrid_model_data(simdir, target_grid, regridder, month_name)

        T_section = extract_midlatitude_section(T_target; apply_dz_weighting)
        S_section = extract_midlatitude_section(S_target; apply_dz_weighting)

        if zonal_average
            T_section = nanmean(T_section, dims=1)
            S_section = nanmean(S_section, dims=1)
        end

        month_output = vcat(T_section[.!isnan.(T_section)], S_section[.!isnan.(S_section)])
        push!(monthly_outputs, month_output)
    end

    # Concatenate all months into single vector
    return vcat(monthly_outputs...)
end

"""
    build_observation_covariance(obs_paths, zonal_average; model_error_frac=0.05)

Build observation covariance from multiple years of monthly ECCO data.

Each year contributes one sample: all 12 months concatenated into a single vector.
This matches the format of the model output (12 monthly averages from year 2).

Note: Covariance is computed WITHOUT dz weighting. The dz weighting is applied
separately when computing Y_target and model forward runs.

Uses SVD-based rank reduction for the internal covariance plus a diagonal
model error term.

Arguments:
- `obs_paths`: Vector of paths to observation data directories
- `zonal_average`: Whether to use zonal averaging
- `model_error_frac`: Fraction of mean field values to use as model error (default 0.05 = 5%)
- `error_regularizer`: Small regularization term added to diagonal for numerical stability (default 1e-6)
"""
function build_observation_covariance(obs_paths, zonal_average; model_error_frac=0.05, error_regularizer=1e-6)
    # Collect yearly observations (all 12 months concatenated per year)
    # No dz weighting applied here - weighting is applied to Y_target and model output
    all_yearly_obs = []
    for obs_path in obs_paths
        monthly_obs = process_monthly_observations(obs_path, zonal_average; apply_dz_weighting=false)
        # Concatenate all 12 months into single vector
        yearly_vec = vcat([monthly_obs[:, m] for m in 1:12]...)
        push!(all_yearly_obs, yearly_vec)
    end

    Y = hcat(all_yearly_obs...)

    @info "Building covariance from $(size(Y, 2)) yearly samples, output dimension = $(size(Y, 1))"

    # SVD-based internal covariance (rank n_trials-1)
    internal_cov = tsvd_cov_from_samples(Y)

    # Model error: fraction of mean field values (diagonal)
    @info "Using model error fraction: $(model_error_frac * 100)%"
    data_mean = vec(mean(Y, dims=2))
    model_error_cov = Diagonal((model_error_frac * data_mean).^2)
    model_error_cov += error_regularizer * I  # Regularization

    # Combine internal variability and model error
    covariance = SVDplusD(internal_cov, model_error_cov)

    return covariance, Y
end

"""
    build_diagonal_covariance(Y_target; T_variance=1.0, S_variance=0.01)

Build a simple diagonal observation covariance with user-specified variances
for temperature and salinity fields.

Assumes Y_target is structured as 12 months of [T..., S...] with equal-sized
T and S sections per month.

Arguments:
- `Y_target`: Flattened observation vector (used to determine dimensions)
- `T_variance`: Variance for temperature observations (default 1.0 °C²)
- `S_variance`: Variance for salinity observations (default 0.25 PSU²)
"""
function build_diagonal_covariance(Y_target; T_variance=1, S_variance=0.25^2)
    n_total = length(Y_target)
    n_per_month = n_total ÷ 12
    n_field = n_per_month ÷ 2  # T and S have equal sizes

    # Build diagonal: 12 months of [T..., S...]
    diagonal_values = repeat(vcat(fill(T_variance, n_field), fill(S_variance, n_field)), 12)

    return Diagonal(diagonal_values)
end