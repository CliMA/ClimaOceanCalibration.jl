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

dataset = ECCO4Monthly()

dir = joinpath(homedir(), "ECCO_data")
mkpath(dir)
start_dates = [DateTime(1992, 1, 1), DateTime(2002, 1, 1)]

buoyancy_model = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())

for start_date in start_dates
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

    SAVE_PATH = joinpath(pwd(), "calibration_data", "ECCO4Monthly", dirname)
    mkpath(SAVE_PATH)

    T_filepath = joinpath(SAVE_PATH, "T.jld2")
    S_filepath = joinpath(SAVE_PATH, "S.jld2")
    b_filepath = joinpath(SAVE_PATH, "b.jld2")

    save_averaged_fieldtimeseries(T_averaged_fts, T, filename=T_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(S_averaged_fts, S, filename=S_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(b_averaged_fts, nothing, filename=b_filepath, overwrite_existing=true)
end

