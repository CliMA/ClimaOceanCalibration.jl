using Test
using Oceananigans
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.OutputReaders
using Oceananigans.OutputReaders: FieldTimeSeries
using ClimaOceanCalibration.DataWrangling: TimeAverageOperator

@testset "TimeAverageOperator for FieldTimeSeries" begin
    # Create a simple grid
    grid = RectilinearGrid(
        size = (4, 4, 4),
        x = (0, 1),
        y = (0, 1),
        z = (0, 1)
    )

    # Create a FieldTimeSeries with uniform time spacing
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create fields with known values for easy testing
    fields = []
    for t in 1:length(times)
        field = CenterField(grid)
        # Make each field contain its time index as value
        set!(field, (x, y, z) -> t)
        push!(fields, field)
    end
    
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)

    for (t, time) in enumerate(times)
        set!(fts[t], (x, y, z) -> x + y + z + t)
    end
    
    @testset "nsteps = 1 (no averaging)" begin
        operator = TimeAverageOperator(fts, 1)
        result = operator(fts)
        
        # With nsteps=1, should return the original time series
        @test length(result.times) == length(fts.times)
        for i in 1:length(times)
            @test all(interior(result[i]) .≈ interior(fts[i]))
        end
    end
    
    @testset "nsteps = 2 (average pairs)" begin
        operator = TimeAverageOperator(fts, 2)
        result = operator(fts)
        
        # Should have half as many times
        @test length(result.times) == 3
        @test result.times == [0.0, 2.0, 4.0]
        
        # Test the averaging results at each grid point
        for i in 1:size(grid, 1)
            for j in 1:size(grid, 2)
                for k in 1:size(grid, 3)
                    # Get grid point coordinates
                    x, y, z = grid.xᶜᵃᵃ[i], grid.yᵃᶜᵃ[j], grid.z.cᵃᵃᶜ[k]
                    
                    # First averaged field: average of fields 1 and 2
                    # (x+y+z+1 + x+y+z+2)/2 = x+y+z+1.5
                    @test interior(result[1])[i, j, k] ≈ x + y + z + 1.5
                    
                    # Second averaged field: average of fields 3 and 4
                    # (x+y+z+3 + x+y+z+4)/2 = x+y+z+3.5
                    @test interior(result[2])[i, j, k] ≈ x + y + z + 3.5
                    
                    # Third averaged field: average of fields 5 and 6
                    # (x+y+z+5 + x+y+z+6)/2 = x+y+z+5.5
                    @test interior(result[3])[i, j, k] ≈ x + y + z + 5.5
                end
            end
        end
    end
    
    @testset "nsteps = 3 (average triplets)" begin
        operator = TimeAverageOperator(fts, 3)
        result = operator(fts)
        
        # Should have 1/3 as many times (rounded down)
        @test length(result.times) == 2
        @test result.times == [0.0, 3.0]
        
        # Test the averaging results at each grid point
        for i in 1:size(grid, 1)
            for j in 1:size(grid, 2)
                for k in 1:size(grid, 3)
                    # Get grid point coordinates
                    x, y, z = grid.xᶜᵃᵃ[i], grid.yᵃᶜᵃ[j], grid.z.cᵃᵃᶜ[k]
                    
                    # First averaged field: average of fields 1, 2, and 3
                    # (x+y+z+1 + x+y+z+2 + x+y+z+3)/3 = x+y+z+2
                    @test interior(result[1])[i, j, k] ≈ x + y + z + 2.0
                    
                    # Second averaged field: average of fields 4, 5, and 6
                    # (x+y+z+4 + x+y+z+5 + x+y+z+6)/3 = x+y+z+5
                    @test interior(result[2])[i, j, k] ≈ x + y + z + 5.0
                end
            end
        end
    end
    
    @testset "Non-uniform time steps should error" begin
        # Create a FieldTimeSeries with non-uniform time spacing
        non_uniform_times = [0.0, 1.0, 3.0, 6.0, 10.0]
        non_uniform_fts = FieldTimeSeries{Center, Center, Center}(grid, non_uniform_times)
        
        # Should throw an assertion error for non-uniform time steps
        @test_throws AssertionError TimeAverageOperator(non_uniform_fts, 2)
    end
end

# using Dates
# using Oceananigans
# using Oceananigans.Fields
# using Oceananigans.Grids
# using ClimaOcean
# using ClimaOcean.Copernicus
# using ClimaOcean.DataWrangling
# using ClimaOcean.DataWrangling
# using ClimaOcean.DataWrangling: download_dataset, NearestNeighborInpainting
# using ClimaOceanCalibration.DataWrangling: TimeAverageOperator
# using PythonCall
# using Statistics

# @testset "TimeAverageOperator Integration Tests" begin
#     # Setup FieldTimeSeries with GLORYS data
#     Nx, Ny, Nz = (360, 180, 60)
#     depth = 6000
#     z_faces = ExponentialCoordinate(Nz, -depth, 0)

#     arch = CPU()
#     grid = TripolarGrid(arch;
#                         size = (Nx, Ny, Nz),
#                            z = z_faces,
#                         halo = (7, 7, 7))

#     bottom_height = regrid_bathymetry(grid; minimum_depth = 15, major_basins = 1, interpolation_passes = 75)
#     grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

#     dataset = GLORYSMonthly()

#     dir = "./GLORYS_data"
#     mkpath(dir)

#     start_date = DateTime(1993, 1, 1)
#     end_date = DateTime(1993, 3, 1)

#     augmented_time = start_date:Month(1):end_date + Month(1)
#     time_diffs = Dates.value.(diff(augmented_time)) ./ 1000  # in seconds

#     T = Metadata(:temperature; dataset, dir, start_date, end_date)
#     inpainting = NearestNeighborInpainting(500)
#     data = FieldTimeSeries(T, grid; inpainting)
#     Nt_data = length(data.times)

#     for nstep in [1, 2, 3]
#         @info "nstep = $nstep"
#         operator = TimeAverageOperator(data, nstep)
#         averaged_data = operator(data)

#         @test length(operator.source_times) == Nt_data
#         @test length(operator.source_Δt) == Nt_data
#         @test length(operator.target_times) == Nt_data ÷ nstep
#         @test length(operator.target_Δt) == Nt_data ÷ nstep

#         if nstep == 1
#             @test all(averaged_data[1].data .≈ data[1].data)
#         else
#             first_mean_value = sum(interior(data[i]) .* time_diffs[i] for i in 1:nstep) ./ Dates.value(augmented_time[1+nstep] - augmented_time[1]) .* 1000
#             @test all(interior(averaged_data[1])[first_mean_value .!== NaN] .≈ first_mean_value[first_mean_value .!== NaN])
#         end

#         @test length(averaged_data) == Nt_data ÷ nstep

#     end
# end