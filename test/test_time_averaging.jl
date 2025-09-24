using Dates
using Oceananigans
using Oceananigans.Fields
using Oceananigans.Grids
using ClimaOcean
using ClimaOcean.Copernicus
using ClimaOcean.DataWrangling
using ClimaOcean.DataWrangling
using ClimaOcean.DataWrangling: download_dataset, NearestNeighborInpainting
using ClimaOceanCalibration.DataWrangling: TimeAverageOperator
using PythonCall
using Statistics

@testset "TimeAverageOperator Integration Tests" begin
    # Setup FieldTimeSeries with GLORYS data
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

    dir = "./GLORYS_data"
    mkpath(dir)

    start_date = DateTime(1993, 1, 1)
    end_date = DateTime(1993, 6, 1)

    augmented_time = start_date:Month(1):end_date + Month(1)
    time_diffs = Dates.value.(diff(augmented_time)) ./ 1000  # in seconds

    T = Metadata(:temperature; dataset, dir, start_date, end_date)
    inpainting = NearestNeighborInpainting(500)
    data = FieldTimeSeries(T, grid; inpainting)
    Nt_data = length(data.times)

    for nstep in [1, 3, 4, 6]
        @info "nstep = $nstep"
        operator = TimeAverageOperator(data, nstep)
        averaged_data = operator(data)

        @test length(operator.source_times) == Nt_data
        @test length(operator.source_Δt) == Nt_data
        @test length(operator.target_times) == Nt_data ÷ nstep
        @test length(operator.target_Δt) == Nt_data ÷ nstep

        if nstep == 1
            @test all(averaged_data[1].data .≈ data[1].data)
        else
            first_mean_value = sum(interior(data[i]) .* time_diffs[i] for i in 1:nstep) ./ Dates.value(augmented_time[1+nstep] - augmented_time[1]) .* 1000
            @test all(interior(averaged_data[1])[first_mean_value .!== NaN] .≈ first_mean_value[first_mean_value .!== NaN])
        end

        @test length(averaged_data) == Nt_data ÷ nstep

    end
end