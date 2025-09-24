using Test
using ClimaOceanCalibration
using ClimaOceanCalibration.DiffusiveOceanCalibration

@testset "ClimaOceanCalibration.jl" begin
    
    @testset "Package Loading" begin
        # Test that the package loads without errors
        @test isdefined(ClimaOceanCalibration, :DiffusiveOceanCalibration)
        @test isdefined(ClimaOceanCalibration, :DataWrangling)
    end

    # Preparing dataset for testing
    include("download_glorys_monthly_figshare_api.jl")
    include("test_time_averaging.jl")
end
