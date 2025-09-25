# Final Working BilinearInterpolator with Full ESMF/xesmf Compatibility

using CondaPkg
using Oceananigans.Grids: AbstractGrid, φnodes, λnodes
using Oceananigans
using SparseArrays
using LinearAlgebra

CondaPkg.add(["pandas", "numpy", "scipy", "xarray", "esmpy", "xesmf"], channel="conda-forge")
CondaPkg.resolve()
using PythonCall

struct BilinearInterpolator{W1, W2}
    set1 :: W1
    set2 :: W2
end

function BilinearInterpolator(grid1::AbstractGrid, grid2::AbstractGrid) 
    W1 = regridder_weights(grid1, grid2; method="bilinear")
    W2 = regridder_weights(grid2, grid1; method="bilinear")
    return BilinearInterpolator(W1, W2)
end

regrid!(dst, weights, scr) = LinearAlgebra.mul!(vec(dst), weights, vec(scr))

get_numpy()  = pyimport("numpy")
get_xarray() = pyimport("xarray")
get_xesmf() = pyimport("xesmf")

two_dimensionalize(lat::Matrix, lon::Matrix) = lat, lon

function two_dimensionalize(lat::AbstractVector, lon::AbstractVector) 
    Nx = length(lon)
    Ny = length(lat)
    lat = repeat(lat', Nx)
    lon = repeat(lon, 1, Ny)
    return lat, lon
end

function coordinate_dataset(grid::AbstractGrid)
    lat = Array(φnodes(grid, Center(), Center(), Center()))
    lon = Array(λnodes(grid, Center(), Center(), Center()))

    lat_b = Array(φnodes(grid, Face(), Face(), Center()))
    lon_b = Array(λnodes(grid, Face(), Face(), Center()))

    lat,   lon   = two_dimensionalize(lat,   lon)
    lat_b, lon_b = two_dimensionalize(lat_b, lon_b)

    return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
end

function structured_coordinate_dataset(lat, lon, lat_b, lon_b)
    numpy  = get_numpy()
    xarray = get_xarray()

    # Transpose in Julia before converting to Python
    lat_t = lat'
    lon_t = lon'
    lat_b_t = lat_b'
    lon_b_t = lon_b'

    # Convert Julia arrays to Python numpy arrays
    py_lat = numpy.array(lat)
    py_lon = numpy.array(lon)
    py_lat_b = numpy.array(lat_b)
    py_lon_b = numpy.array(lon_b)
    
    py_lat_t = numpy.array(lat_t)
    py_lon_t = numpy.array(lon_t)
    py_lat_b_t = numpy.array(lat_b_t)
    py_lon_b_t = numpy.array(lon_b_t)

    # Create coordinate dictionaries
    lat_coords = Dict(
        "lat" => (["y", "x"], py_lat_t),
        "lon" => (["y", "x"], py_lon_t)
    )
    
    lon_coords = Dict(
        "lat" => (["y", "x"], py_lat_t),
        "lon" => (["y", "x"], py_lon_t)
    )
    
    lat_b_coords = Dict(
        "lat_b" => (["y_b", "x_b"], py_lat_b_t),
        "lon_b" => (["y_b", "x_b"], py_lon_b_t)
    )
    
    lon_b_coords = Dict(
        "lat_b" => (["y_b", "x_b"], py_lat_b_t),
        "lon_b" => (["y_b", "x_b"], py_lon_b_t)
    )

    # Create DataArrays
    ds_lat = xarray.DataArray(
        py_lat_t,
        dims=["y", "x"],
        coords=lat_coords,
        name="latitude"
    )
    
    ds_lon = xarray.DataArray(
        py_lon_t,
        dims=["y", "x"],
        coords=lon_coords,
        name="longitude"
    )
    
    ds_lat_b = xarray.DataArray(
        py_lat_b_t,
        dims=["y_b", "x_b"],
        coords=lat_b_coords
    )

    ds_lon_b = xarray.DataArray(
        py_lon_b_t,
        dims=["y_b", "x_b"],
        coords=lon_b_coords
    )

    # Create Dataset
    dataset_dict = Dict(
        "lat"   => ds_lat, 
        "lon"   => ds_lon,
        "lat_b" => ds_lat_b,
        "lon_b" => ds_lon_b
    )

    return xarray.Dataset(dataset_dict)
end

function regridder_weights(dst::AbstractGrid, src::AbstractGrid; method::String="bilinear")
    # Test imports with automatic fixes
    
    src_ds = coordinate_dataset(src)
    dst_ds = coordinate_dataset(dst)

    # Create regridder with periodic boundary conditions
    regridder = get_xesmf().Regridder(src_ds, dst_ds, method, periodic=true) 

    # Extract weights data
    weights_data = regridder.weights.data
    
    # Convert Python objects to Julia types using pyconvert
    shape = pyconvert(Tuple, weights_data.shape)
    vals = pyconvert(Vector{Float64}, weights_data.data)
    coords = pyconvert(Array, weights_data.coords)
    
    # Extract row and column indices (convert from 0-based to 1-based indexing)
    rows = pyconvert(Vector{Int}, coords[1, :]) .+ 1
    cols = pyconvert(Vector{Int}, coords[2, :]) .+ 1

    # Create Julia sparse matrix
    W = sparse(rows, cols, vals, shape[1], shape[2])

    return W
end