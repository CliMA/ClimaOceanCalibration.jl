using Test
using ClimaOceanCalibration
using ClimaOceanCalibration.DiffusiveOceanCalibration

@testset "ClimaOceanCalibration.jl" begin
    @testset "Package Loading" begin
        # Test that the package loads without errors
        @test isdefined(ClimaOceanCalibration, :DiffusiveOceanCalibration)
        @test isdefined(ClimaOceanCalibration, :DataWrangling)
    end

    # Include the download script but don't run it automatically
    # include("download_glorys_monthly_figshare_api.jl")
    
    # # Check if data directory exists and has files
    # data_dir = "./GLORYS_data"
    # if !isdir(data_dir) || isempty(readdir(data_dir))
    #     # Only download if directory doesn't exist or is empty
    #     download_from_figshare_api(figshare_id, data_dir)
    # end
    
    include("test_time_averaging.jl")
end
