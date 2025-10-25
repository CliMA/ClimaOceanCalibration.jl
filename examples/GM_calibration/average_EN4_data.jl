using ClimaOcean
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using SeawaterPolynomials.TEOS10
using ClimaOcean.DataWrangling
using Printf
using Dates
using CUDA
using ClimaOceanCalibration.DataWrangling
using JLD2
using XESMF

arch = GPU()

grid = jldopen(joinpath(pwd(), "examples", "GM_calibration", "grids_and_regridder.jld2"), "r") do file
    return on_architecture(arch, file["target_grid"])
end

dataset = EN4Monthly()

dir = joinpath(homedir(), "EN4_data")
mkpath(dir)
start_dates = [DateTime(1902), DateTime(1912), DateTime(1922), DateTime(1942),
               DateTime(1952), DateTime(1972),
               DateTime(1992), DateTime(2002), DateTime(2012)]

# seems that T fields for 1939 1971 1985 is problematic

buoyancy_model = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())

for start_date in start_dates
    @info "Processing data starting from $(start_date)..."
    end_date = start_date + Year(10) - Month(1)

    T = Metadata(:temperature; dataset, dir, start_date, end_date)
    S = Metadata(:salinity; dataset, dir, start_date, end_date)

    T_data = FieldTimeSeries(T, grid, time_indices_in_memory=20)
    S_data = FieldTimeSeries(S, grid, time_indices_in_memory=20)

    T_averaging = TimeAverageOperator(T_data)
    T_averaged_fts = AveragedFieldTimeSeries(T_averaging(T_data), T_averaging, nothing)

    S_averaging = TimeAverageOperator(S_data)
    S_averaged_fts = AveragedFieldTimeSeries(S_averaging(S_data), S_averaging, nothing)

    b_averaging = TimeAverageBuoyancyOperator(T_data)
    b_averaged_fts = AveragedFieldTimeSeries(b_averaging(T_data, S_data, buoyancy_model), b_averaging, nothing)

    prefix = "10yearaverage_2degree"
    date_str = replace(string(start_date), ":" => "-")

    dirname = prefix * date_str

    SAVE_PATH = joinpath(pwd(), "calibration_data", "EN4Monthly", dirname)
    mkpath(SAVE_PATH)

    T_filepath = joinpath(SAVE_PATH, "T.jld2")
    S_filepath = joinpath(SAVE_PATH, "S.jld2")
    b_filepath = joinpath(SAVE_PATH, "b.jld2")

    save_averaged_fieldtimeseries(T_averaged_fts, T, filename=T_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(S_averaged_fts, S, filename=S_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(b_averaged_fts, nothing, filename=b_filepath, overwrite_existing=true)
end

# seems that T fields for 1939 1971 1985 is problematic
start_dates = [DateTime(1902), DateTime(1907), DateTime(1912), DateTime(1917), DateTime(1922),
               DateTime(1927), DateTime(1932), DateTime(1942),
               DateTime(1947), DateTime(1952), DateTime(1957), DateTime(1962),
               DateTime(1972), DateTime(1977),
               DateTime(1987), DateTime(1992), DateTime(1997), DateTime(2002),
               DateTime(2007), DateTime(2012)]

for start_date in start_dates
    @info "Processing data starting from $(start_date)..."
    end_date = start_date + Year(5) - Month(1)

    T = Metadata(:temperature; dataset, dir, start_date, end_date)
    S = Metadata(:salinity; dataset, dir, start_date, end_date)

    T_data = FieldTimeSeries(T, grid, time_indices_in_memory=20)
    S_data = FieldTimeSeries(S, grid, time_indices_in_memory=20)

    T_averaging = TimeAverageOperator(T_data)
    T_averaged_fts = AveragedFieldTimeSeries(T_averaging(T_data), T_averaging, nothing)

    S_averaging = TimeAverageOperator(S_data)
    S_averaged_fts = AveragedFieldTimeSeries(S_averaging(S_data), S_averaging, nothing)

    b_averaging = TimeAverageBuoyancyOperator(T_data)
    b_averaged_fts = AveragedFieldTimeSeries(b_averaging(T_data, S_data, buoyancy_model), b_averaging, nothing)

    prefix = "5yearaverage_2degree"
    date_str = replace(string(start_date), ":" => "-")

    dirname = prefix * date_str

    SAVE_PATH = joinpath(pwd(), "calibration_data", "EN4Monthly", dirname)
    mkpath(SAVE_PATH)

    T_filepath = joinpath(SAVE_PATH, "T.jld2")
    S_filepath = joinpath(SAVE_PATH, "S.jld2")
    b_filepath = joinpath(SAVE_PATH, "b.jld2")

    save_averaged_fieldtimeseries(T_averaged_fts, T, filename=T_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(S_averaged_fts, S, filename=S_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(b_averaged_fts, nothing, filename=b_filepath, overwrite_existing=true)
end

