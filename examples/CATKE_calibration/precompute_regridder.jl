using NumericalEarth
using Oceananigans
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CUDA
using ConservativeRegridding
using JLD2
using KernelAbstractions: @index, @kernel

Nz = 100
z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
Nx_target, Ny_target = (90, 42)

minimum_depth = 15
major_basins = 1
interpolation_passes = 55

arch = GPU()
Nx_source, Ny_source = (720, 360)
source_grid = TripolarGrid(arch;
                           size = (Nx_source, Ny_source, Nz),
                           z = z_faces,
                           halo = (7, 7, 7))

bottom_height_source = regrid_bathymetry(source_grid; minimum_depth, major_basins, interpolation_passes)
source_grid = ImmersedBoundaryGrid(source_grid, GridFittedBottom(bottom_height_source); active_cells_map=true)

target_grid = LatitudeLongitudeGrid(arch; size=(Nx_target, Ny_target, Nz), z = z_faces,
                                    longitude=(0, 360), latitude=(-84, 84))

@kernel function _find_immersed_height!(bottom_height, grid, field)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    kmax = 0
    @inbounds for k in 1:Nz
        kmax = ifelse(isnan(field[i, j, k]), k, kmax)
    end

    @inbounds bottom_height[i, j, 1] = ifelse(kmax == 0, grid.z.cᵃᵃᶠ[1], grid.z.cᵃᵃᶜ[kmax])
end

function find_immersed_height!(bottom_height, grid, field)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _find_immersed_height!, bottom_height, grid, field)
    return nothing
end

src_field = CenterField(source_grid)
mask_immersed_field!(src_field, NaN)

dst_field = CenterField(target_grid)

# The regridder carries horizontal weights only, and is built from single-level,
# vertically regular copies of the grids: that is what pairs each cell of the
# tripolar fold row with its partner instead of counting it twice.
regridder_source_grid = TripolarGrid(size=(Nx_source, Ny_source, 1), z=(0, 1), halo=(7, 7, 1))

regridder_target_grid = LatitudeLongitudeGrid(size=(Nx_target, Ny_target, 1), z=(0, 1),
                                              longitude=(0, 360), latitude=(-84, 84))

regridder = ConservativeRegridding.Regridder(regridder_target_grid, regridder_source_grid)
regridder = on_architecture(arch, regridder)

for k in 1:Nz
    ConservativeRegridding.regrid!(vec(interior(dst_field, :, :, k)), regridder,
                                   vec(interior(src_field, :, :, k)))
end

fill_halo_regions!(dst_field)

bottom_height_target = Field{Center, Center, Nothing}(target_grid)
find_immersed_height!(bottom_height_target, target_grid, dst_field)

target_grid = ImmersedBoundaryGrid(target_grid, GridFittedBottom(bottom_height_target); active_cells_map=true)

new_field = CenterField(target_grid)
mask_immersed_field!(new_field, NaN)
@assert sum(isnan.(interior(new_field))) == sum(isnan.(interior(dst_field)))

SAVE_PATH = joinpath(pwd(), "examples", "CATKE_calibration", "4deg_grids_and_regridder.jld2")
jldopen(SAVE_PATH, "w") do file
    file["source_grid"] = on_architecture(CPU(), source_grid)
    file["target_grid"] = on_architecture(CPU(), target_grid)
    file["regridder"] = on_architecture(CPU(), regridder)
end