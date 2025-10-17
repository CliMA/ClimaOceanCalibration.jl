using ClimaOcean
using Oceananigans
using Oceananigans.Architectures: on_architecture
using CUDA
using XESMF
using JLD2

import Oceananigans.Architectures: on_architecture

Nz = 100
z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
Nx_target, Ny_target = (180, 84)

minimum_depth = 15
major_basins = 1
interpolation_passes = 55

arch = GPU()
target_grid = LatitudeLongitudeGrid(arch; size=(Nx_target, Ny_target, Nz), z = z_faces,
                                    longitude=(0, 360), latitude=(-84, 84))

bottom_height_target = regrid_bathymetry(target_grid; minimum_depth, major_basins, interpolation_passes)
target_grid = ImmersedBoundaryGrid(target_grid, GridFittedBottom(bottom_height_target); active_cells_map = true)

Nx_source, Ny_source = (720, 360)
source_grid = TripolarGrid(arch;
                           size = (Nx_source, Ny_source, Nz),
                           z = z_faces,
                           halo = (7, 7, 7))

bottom_height_source = regrid_bathymetry(source_grid; minimum_depth, major_basins, interpolation_passes)
source_grid = ImmersedBoundaryGrid(source_grid, GridFittedBottom(bottom_height_source); active_cells_map=true)

src_field = CenterField(source_grid)
dst_field = CenterField(target_grid)

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

on_architecture(on, r::XESMF.Regridder) = XESMF.Regridder(on_architecture(on, r.method),
                                                          on_architecture(on, r.weights),
                                                          on_architecture(on, r.src_temp),
                                                          on_architecture(on, r.dst_temp))

jldopen("grids_and_regridder.jld2", "w") do file
    file["source_grid"] = on_architecture(CPU(), source_grid)
    file["target_grid"] = on_architecture(CPU(), target_grid)
    file["regridder"] = on_architecture(CPU(), regridder)
end