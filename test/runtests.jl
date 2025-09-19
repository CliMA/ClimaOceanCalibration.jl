using Test
using ClimaOceanCalibration
using ClimaOceanCalibration.DiffusiveOceanCalibration

@testset "ClimaOceanCalibration.jl" begin
    
    @testset "Package Loading" begin
        # Test that the package loads without errors
        @test isdefined(ClimaOceanCalibration, :DiffusiveOceanCalibration)
    end
end
