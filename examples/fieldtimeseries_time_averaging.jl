using ClimaOcean
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids
using ClimaOcean.Copernicus
using ClimaOcean.DataWrangling
using Printf
using Dates
using PythonCall
using ClimaOcean.DataWrangling: download_dataset, NearestNeighborInpainting
using CUDA
using ClimaOceanCalibration.DataWrangling: TimeAverageOperator

Nx, Ny, Nz = (360, 180, 60)
depth = 6000
z_faces = ExponentialCoordinate(Nz, -depth, 0)

arch = CPU()
grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; minimum_depth = 15, major_basins = 1, interpolation_passes = 75)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

dataset = GLORYSMonthly()

dir = joinpath(pwd(), "GLORYS_data")
mkpath(dir)

start_date = DateTime(1993, 1, 1)
end_date = DateTime(1993, 6, 1)

T = Metadata(:temperature; dataset, dir, start_date, end_date)

inpainting = NearestNeighborInpainting(500)
T_data = FieldTimeSeries(T, grid; inpainting)

averaging_operator = TimeAverageOperator(T_data, 4)
averaged_fts = averaging_operator(T_data)