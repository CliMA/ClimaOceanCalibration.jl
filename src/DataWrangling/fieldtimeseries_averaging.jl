using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.OutputReaders: FieldTimeSeries
using ClimaOcean
using ClimaOcean.DataWrangling: DatasetFieldTimeSeries
using Dates
using JLD2

"""
    AveragedFieldTimeSeries{D, T, S}

A container for field data that has been averaged in time and/or space.

# Fields
- `data`: The averaged field time series data
- `time_averaging`: Information about the time averaging operation applied
- `space_averaging`: Information about the space averaging operation applied

This struct provides a way to track both the averaged data and the operations used to produce it.
"""
struct AveragedFieldTimeSeries{D, T, S}
    data            :: D
    time_averaging  :: T
    space_averaging :: S
end

"""
    TimeAverageOperator{N, ST, TT, SDT, TDT}

An operator that performs time averaging on field time series data.

# Fields
- `nsteps`: Number of time steps to combine in each averaging window
- `source_times`: Original times from the source data
- `target_times`: Times for the averaged data (subset of source times)
- `source_Δt`: Time intervals in the source data
- `target_Δt`: Time intervals in the averaged data

This operator is used to reduce temporal resolution by averaging multiple time steps together.
"""
struct TimeAverageOperator{N, ST, TT, SDT, TDT}
    nsteps       :: N
    source_times :: ST
    target_times :: TT
    source_Δt    :: SDT
    target_Δt    :: TDT
end

floor_multiple(a, b) = a - rem(a, b)

"""
    TimeAverageOperator(fts::DatasetFieldTimeSeries, nsteps)

Create a time averaging operator that averages every `nsteps` time steps in the field time series.
Note that the assumption is that fts[i] is the average field value over the interval [times[i], times[i+1]].
For the last timestep, we assume it is averaged over the interval [times[end], times[end] + Δt], where Δt is the date step (which depends on the actual dates given by the metadata).

# Arguments
- `fts`: A `DatasetFieldTimeSeries` containing the time data to be averaged
- `nsteps`: Number of consecutive time steps to average together

# Returns
- `TimeAverageOperator` that can be applied to a compatible field time series

# Notes
- If `nsteps` is 1, no averaging will be performed
- The operator computes target times and appropriate time intervals for weighted averaging
- For dataset time series with dates, proper date-based time intervals are calculated
"""
function TimeAverageOperator(fts::DatasetFieldTimeSeries, nsteps)
    fts.times isa Number && return TimeAverageOperator(1, nothing)

    source_dates = fts.backend.metadata.dates
    source_datestep = source_dates |> step
    source_enddate = last(source_dates) + source_datestep
    
    fts_times = Array(fts.times)
    last_timestep = Dates.value(source_enddate - first(source_dates)) / 1000

    times_inclusive = vcat(fts_times, last_timestep)
    source_Δt = diff(times_inclusive)

    truncated_length = floor_multiple(length(fts_times), nsteps)
    target_times = fts_times[1:truncated_length][1:nsteps:end]
    target_Δt = diff(times_inclusive[1:truncated_length+1][1:nsteps:end])

    return TimeAverageOperator(nsteps, fts_times, target_times, source_Δt, target_Δt)
end

"""
    TimeAverageOperator(fts::FieldTimeSeries, nsteps)

Create a time averaging operator that averages every `nsteps` time steps in a regular field time series.
The assumption is that fts[i] is the average field value over the interval [times[i], times[i+1]].
For the last timestep, we assume it extends one timestep beyond the final recorded time.

# Arguments
- `fts`: A `FieldTimeSeries` containing the time data to be averaged
- `nsteps`: Number of consecutive time steps to average together

# Returns
- `TimeAverageOperator` that can be applied to a compatible field time series

# Notes
- If `nsteps` is 1, no averaging will be performed
- The operator requires uniform time spacing in the input field time series
- The operator truncates the data to ensure complete averaging windows
- The returned operator contains both source and target times and time intervals needed for weighted averaging

# Throws
- Assertion error if non-uniform time steps are detected in the input field time series
"""
function TimeAverageOperator(fts::FieldTimeSeries, nsteps)
    fts.times isa Number && return TimeAverageOperator(1, nothing)

    fts_times = Array(fts.times)
    timestep = fts_times[2] - fts_times[1] # assume uniform spacing!!
    if length(fts_times) > 2
        all_timesteps = diff(fts_times)
        @assert all(isapprox.(all_timesteps, timestep)) "Non-uniform time steps detected in FieldTimeSeries. This implementation requires uniform time spacing."
    end

    last_timestep = fts_times[end] + timestep

    times_inclusive = vcat(fts_times, last_timestep)
    source_Δt = diff(times_inclusive)

    truncated_length = floor_multiple(length(fts_times), nsteps)
    target_times = fts_times[1:truncated_length][1:nsteps:end]
    target_Δt = diff(times_inclusive[1:truncated_length+1][1:nsteps:end])

    return TimeAverageOperator(nsteps, fts_times, target_times, source_Δt, target_Δt)
end

TimeAverageOperator(fts) = TimeAverageOperator(fts, length(fts))

"""
    (𝒯::TimeAverageOperator)(fts::FieldTimeSeries)

Apply time averaging to a field time series using the specified operator.

# Arguments
- `𝒯`: The time averaging operator
- `fts`: The field time series to which the operator is applied

# Returns
A new field time series with reduced temporal resolution, where each time step is an average of `nsteps` original time steps.

# Example

```julia
# Create a time averaging operator for 3 time steps
operator = TimeAverageOperator(fts, 3)

# Apply the operator to a field time series
averaged_fts = operator(fts)
```

"""
function (𝒯::TimeAverageOperator)(fts::FieldTimeSeries)
    nsteps = 𝒯.nsteps
    nsteps == 1 && return fts

    LX, LY, LZ = location(fts)
    grid = fts.grid
    boundary_conditions = fts.boundary_conditions
    target_fts = FieldTimeSeries{LX, LY, LZ}(grid, 𝒯.target_times; boundary_conditions)

    for i in eachindex(𝒯.target_times)
        target_field = target_fts[i]
        for j in 1:nsteps
            target_field .+= fts[nsteps * (i-1) + j] * 𝒯.source_Δt[nsteps * (i-1) + j]
        end
        target_field ./= 𝒯.target_Δt[i]
    end
    return target_fts
end

function save_averaged_fieldtimeseries(afts::AveragedFieldTimeSeries, metadata; filename::String="averaged_fieldtimeseries", overwrite_existing::Bool=false)
    # add .jld2 to filename if not present
    if !endswith(filename, ".jld2")
        filename *= ".jld2"
    end

    # only save if file doesn't exist or if overwrite_existing is true
    if overwrite_existing || !isfile(filename)
        jldopen(filename, "w+") do file
            file["averaged_fieldtimeseries"] = afts
            file["metadata"] = metadata
        end
    end
    return nothing
end