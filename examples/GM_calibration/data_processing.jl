using Oceananigans
using Oceananigans.Grids: znodes, φnodes
using Oceananigans.Fields: location
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: on_architecture
using XESMF
using JLD2

function regrid_model_data(simdir)
    @info "Regridding model data in $(simdir)..."
    filepath = joinpath(simdir, "ocean_complete_fields_10year_average_calibrationsample.jld2")
    T_data = FieldTimeSeries(filepath, "T", backend=OnDisk())
    S_data = FieldTimeSeries(filepath, "S", backend=OnDisk())

    source_grid = T_data.grid
    LX, LY, LZ = location(T_data)
    boundary_conditions = T_data.boundary_conditions
    times = T_data.times

    Nx, Ny, Nz = (180, 84, 100)
    z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)

    arch = CPU()
    target_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), z = z_faces,
                                longitude=(0, 360), latitude=(-84, 84))

    bottom_height = regrid_bathymetry(target_grid; minimum_depth = 15, major_basins = 1, interpolation_passes = 55)
    target_grid = ImmersedBoundaryGrid(target_grid, GridFittedBottom(bottom_height); active_cells_map = true)

    T_target = FieldTimeSeries{LX, LY, LZ}(grid, times; boundary_conditions)
    S_target = FieldTimeSeries{LX, LY, LZ}(grid, times; boundary_conditions)

    src_field = T_data[1]
    dst_field = T_target[1]

    regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

    for t in 1:length(times)
        regrid!(T_target[t], regridder, T_data[t])
        regrid!(S_target[t], regridder, S_data[t])
        mask_immersed_field!(T_target[t], NaN)
        mask_immersed_field!(S_target[t], NaN)
    end
    return T_target, S_target
end

taper_interior_ocean(z, z_scale=3500, width=1000) = 0.5 * (1 + tanh((z + z_scale) / width))
no_tapering(z) = 1

function extract_field_section(fts::FieldTimeSeries, latitude_range; vertical_weighting=no_tapering)
    fts = on_architecture(CPU(), fts)
    LX, LY, LZ = location(fts)
    grid = fts.grid

    φᶜ = φnodes(grid, LX(), LY(), LZ())
    zᶜ = znodes(grid, LX(), LY(), LZ())
    φmin, φmax = latitude_range

    lat_indices = findfirst(x -> x >= φmin, φᶜ):findlast(x -> x <= φmax, φᶜ)
    z_weights = vertical_weighting.(zᶜ)

    times = fts.times

    Nt = length(times)
    for t in 1:length(times)
        mask_immersed_field!(fts[t], NaN)
    end

    field_section = reshape(z_weights, 1, 1, :) .* interior(fts[Nt], :, lat_indices, :)

    return field_section
end

extract_southern_ocean_section(fts, vertical_weighting=no_tapering) = extract_field_section(fts, (-80, -50); vertical_weighting)

function process_member_data(simdir)
    T_target, S_target = regrid_model_data(simdir)

    T_section = extract_southern_ocean_section(T_target, taper_interior_ocean)
    S_section = extract_southern_ocean_section(S_target, taper_interior_ocean)
    
    return vcat(vec(T_section), vec(S_section))
end