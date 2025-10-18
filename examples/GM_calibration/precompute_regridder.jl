using ClimaOcean
using Oceananigans
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CUDA
using XESMF
using JLD2
using KernelAbstractions: @index, @kernel

import Oceananigans.Architectures: on_architecture

Nz = 100
z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
Nx_target, Ny_target = (180, 84)

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

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

on_architecture(on, r::XESMF.Regridder) = XESMF.Regridder(on_architecture(on, r.method),
                                                          on_architecture(on, r.weights),
                                                          on_architecture(on, r.src_temp),
                                                          on_architecture(on, r.dst_temp))

regrid!(dst_field, regridder, src_field)

bottom_height_target = Field{Center, Center, Nothing}(target_grid)
find_immersed_height!(bottom_height_target, target_grid, dst_field)

target_grid = ImmersedBoundaryGrid(target_grid, GridFittedBottom(bottom_height_target); active_cells_map=true)

new_field = CenterField(target_grid)
mask_immersed_field!(new_field, NaN)
@assert sum(isnan.(interior(new_field))) == sum(isnan.(interior(dst_field)))

SAVE_PATH = joinpath(pwd(), "examples", "GM_calibration", "grids_and_regridder.jld2")
jldopen(SAVE_PATH, "w") do file
    file["source_grid"] = on_architecture(CPU(), source_grid)
    file["target_grid"] = on_architecture(CPU(), target_grid)
    file["regridder"] = on_architecture(CPU(), regridder)
end