using Test
using Dates
using CairoMakie
using Oceananigans
using Oceananigans.Units
using ClimaOceanCalibration.Visualization

@testset "Visualization" begin
    dir = mktempdir()
    grid = LatitudeLongitudeGrid(size = (8, 6, 4), longitude = (0, 360), latitude = (-60, 60), z = (-1000, 0), halo = (5, 5, 5))
    bottom(λ, φ) = -1000 + 800 * exp(-((λ - 180) / 40)^2 - (φ / 20)^2)
    model = HydrostaticFreeSurfaceModel(ImmersedBoundaryGrid(grid, GridFittedBottom(bottom)); tracers = :T)
    set!(model, T = (λ, φ, z) -> 20 + z / 100)
    simulation = Simulation(model, Δt = 10minutes, stop_iteration = 6, verbose = false)
    T = model.tracers.T
    simulation.output_writers[:fields] = JLD2Writer(model, (; T); schedule = IterationInterval(2), dir, filename = "run_fields",
                                                    overwrite_existing = true, file_splitting = IterationInterval(4))
    simulation.output_writers[:surface] = JLD2Writer(model, (; T = view(T, :, :, 4)); schedule = IterationInterval(2), dir,
                                                     filename = "run_surface", overwrite_existing = true)
    simulation.output_writers[:averages] = JLD2Writer(model, (; T_h = Average(T, dims = (1, 2)), T_avg = Average(T));
                                                      schedule = IterationInterval(2), dir, filename = "run_averages", overwrite_existing = true)
    run!(simulation)

    a = Run(dir; name = "a")
    @test keys(a) == ["averages/T_avg", "averages/T_h", "fields/T", "surface/T"]
    @test size(a["fields/T"]) == (8, 6, 4, 4)
    @test size(a["surface/T"]) == (8, 6, 1, 4)
    @test size(a["averages/T_h"]) == (1, 1, 4, 4)

    b = Run("b", Dict("fields/T" => a["fields/T"], "averages/T_avg" => a["averages/T_avg"]))
    for section in (:x, :y, :z)
        @test dashboard(a; fields = keys(a), section) isa Figure
        @test dashboard(a, b; section, reference_date = DateTime(1958, 1, 1)) isa Figure
    end
end
