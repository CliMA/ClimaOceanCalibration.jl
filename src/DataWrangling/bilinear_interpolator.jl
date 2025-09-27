using CondaPkg
using Oceananigans.Grids: AbstractGrid, φnodes, λnodes
using Oceananigans.Fields: AbstractField
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: on_architecture, architecture
using KernelAbstractions: @kernel, @index
using Oceananigans
using SparseArrays
using LinearAlgebra
using CUDA

CondaPkg.add(["pandas", "numpy", "scipy", "xarray", "esmpy", "xesmf"], channel="conda-forge")
CondaPkg.resolve()
using PythonCall

"""
    BilinearInterpolator{M}

A struct for bilinear interpolation between Oceananigans grids using precomputed weights.

# Fields
- `weights :: M`: Sparse weight matrix for interpolation between source and destination grids
"""
struct BilinearInterpolator{M}
    weights :: M
end

"""
    BilinearInterpolator(destination::AbstractGrid, source::AbstractGrid)

Create a bilinear interpolator that computes and stores weights for interpolation from a source grid to a destination grid.

# Arguments
- `destination::AbstractGrid`: The destination grid to interpolate to
- `source::AbstractGrid`: The source grid to interpolate from

# Returns
- `BilinearInterpolator`: An interpolator object with precomputed weights

# Example
```julia
dst_grid = RectilinearGrid(size=(100, 100), x=(0, 1), y=(0, 1))
src_grid = RectilinearGrid(size=(50, 50), x=(0, 1), y=(0, 1))
interpolator = BilinearInterpolator(dst_grid, src_grid)
````
"""
function BilinearInterpolator(destination::AbstractGrid, source::AbstractGrid)
    arch = architecture(destination)
    weights = regridder_weights(destination, source; method="bilinear")
    return BilinearInterpolator(on_architecture(arch, weights))
end

"""
    regrid!(dst, weights, src)

Apply interpolation weights to source data and store the result in the destination array.
This operation performs: vec(dst) = weights * vec(src)

# Arguments
- `dst`: Destination array where results will be stored
- `weights`: Sparse weight matrix for interpolation
- `src`: Source data to be interpolated
"""
regrid!(dst, weights, src) = LinearAlgebra.mul!(vec(dst), weights, vec(src))

function regrid!(dst, src, interpolator::BilinearInterpolator)
    weights = interpolator.weights
    regrid!(dst, weights, src)
end

function regrid!(dst, weights::CuSparseMatrixCSC, src)
    vec(dst) .= weights * CuArray(vec(src))
end

function regrid!(dst::AbstractField, src::AbstractField, interpolator::BilinearInterpolator)
    weights = interpolator.weights
    
    # Get the interior data
    dst_data = interior(dst)
    src_data = interior(src)
    
    Nz = size(src_data, 3)
    
    for k in 1:Nz
        src_slice = view(src_data, :, :, k)
        dst_slice = view(dst_data, :, :, k)
        regrid!(dst_slice, weights, src_slice)
    end
    
    return dst
end

(interpolator::BilinearInterpolator)(destination, source) = regrid!(destination, interpolator.weights, source)

"""
    get_numpy()

Import and return the Python numpy module using PythonCall.
"""
get_numpy()  = pyimport("numpy")

"""
    get_xarray()

Import and return the Python xarray module using PythonCall.
"""
get_xarray() = pyimport("xarray")

"""
    get_xesmf()

Import and return the Python xesmf module using PythonCall.
"""
get_xesmf() = pyimport("xesmf")

"""
    two_dimensionalize(lat::Matrix, lon::Matrix)

Return the input matrices unchanged since they're already two-dimensional.

# Arguments
- `lat::Matrix`: Latitude values as a matrix
- `lon::Matrix`: Longitude values as a matrix

# Returns
- The original latitude and longitude matrices
"""
two_dimensionalize(lat::Matrix, lon::Matrix) = lat, lon

"""
    two_dimensionalize(lat::AbstractVector, lon::AbstractVector)

Convert one-dimensional latitude and longitude vectors into two-dimensional matrices
suitable for use with xesmf regridding.

# Arguments
- `lat::AbstractVector`: Vector of latitude values
- `lon::AbstractVector`: Vector of longitude values

# Returns
- Tuple of two matrices: (2D latitude matrix, 2D longitude matrix)
"""
function two_dimensionalize(lat::AbstractVector, lon::AbstractVector) 
    Nx = length(lon)
    Ny = length(lat)
    lat = repeat(lat', Nx)
    lon = repeat(lon, 1, Ny)
    return lat, lon
end

"""
    coordinate_dataset(grid::AbstractGrid)

Extract coordinate information from an Oceananigans grid and convert it to a format
compatible with xesmf regridding.

# Arguments
- `grid::AbstractGrid`: An Oceananigans grid

# Returns
- Python xarray.Dataset containing grid coordinate information
"""
function coordinate_dataset(grid::AbstractGrid)
    lat = Array(φnodes(grid, Center(), Center(), Center()))
    lon = Array(λnodes(grid, Center(), Center(), Center()))

    lat_b = Array(φnodes(grid, Face(), Face(), Center()))
    lon_b = Array(λnodes(grid, Face(), Face(), Center()))

    lat,   lon   = two_dimensionalize(lat,   lon)
    lat_b, lon_b = two_dimensionalize(lat_b, lon_b)

    return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
end

"""
    structured_coordinate_dataset(lat, lon, lat_b, lon_b)

Create a Python xarray Dataset from latitude and longitude arrays suitable for xesmf regridding.

# Arguments
- `lat`: 2D array of cell-centered latitudes
- `lon`: 2D array of cell-centered longitudes
- `lat_b`: 2D array of cell-boundary latitudes
- `lon_b`: 2D array of cell-boundary longitudes

# Returns
- Python xarray.Dataset containing the structured coordinate information
"""
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

"""
    regridder_weights(dst::AbstractGrid, src::AbstractGrid; method::String="bilinear")

Compute sparse weight matrices for interpolating data between source and destination grids.

# Arguments
- `dst::AbstractGrid`: Destination grid for interpolation
- `src::AbstractGrid`: Source grid for interpolation
- `method::String="bilinear"`: Interpolation method (default: "bilinear")

# Returns
- Sparse matrix of interpolation weights
"""
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