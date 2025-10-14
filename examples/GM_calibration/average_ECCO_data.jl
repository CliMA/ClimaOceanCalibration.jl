using ClimaOcean
using Oceananigans
using Oceananigans.Units
using ClimaOcean.DataWrangling
using Printf
using Dates
using CUDA
using ClimaOceanCalibration.DataWrangling: TimeAverageOperator, AveragedFieldTimeSeries, save_averaged_fieldtimeseries

Nx, Ny, Nz = (180, 84, 100)
z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)

arch = GPU()
grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), z = z_faces,
                            longitude=(0, 360), latitude=(-84, 84))

bottom_height = regrid_bathymetry(grid; minimum_depth = 15, major_basins = 1, interpolation_passes = 55)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

dataset = ECCO4Monthly()

dir = joinpath(homedir(), "ECCO_data")
mkpath(dir)
start_dates = [DateTime(1992, 1, 1), DateTime(2002, 1, 1)]

SAVE_PATH = joinpath(pwd(), "calibration_data", "ECCO4Monthly")
mkpath(SAVE_PATH)

for start_date in start_dates
    start_date = DateTime(2002, 1, 1)
    end_date = start_date + Year(10) - Month(1)

    T = Metadata(:temperature; dataset, dir, start_date, end_date)
    S = Metadata(:salinity; dataset, dir, start_date, end_date)

    T_data = FieldTimeSeries(T, grid, time_indices_in_memory=20)
    S_data = FieldTimeSeries(S, grid, time_indices_in_memory=20)

    T_averaging = TimeAverageOperator(T_data)
    T_averaged_fts = AveragedFieldTimeSeries(T_averaging(T_data), T_averaging, nothing)

    S_averaging = TimeAverageOperator(S_data)
    S_averaged_fts = AveragedFieldTimeSeries(S_averaging(S_data), S_averaging, nothing)

    prefix = "10yearaverage_2degree"
    date_str = replace(string(start_date), ":" => "-")

    T_filename = prefix * "_T_" * date_str
    S_filename = prefix * "_S_" * date_str

    T_filepath = joinpath(SAVE_PATH, T_filename)
    S_filepath = joinpath(SAVE_PATH, S_filename)

    save_averaged_fieldtimeseries(T_averaged_fts, T, filename=T_filepath, overwrite_existing=true)
    save_averaged_fieldtimeseries(S_averaged_fts, S, filename=S_filepath, overwrite_existing=true)
end

