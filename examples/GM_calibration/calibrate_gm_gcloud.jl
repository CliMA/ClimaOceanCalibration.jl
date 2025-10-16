using ClimaCalibrate
using ClimaOceanCalibration.DataWrangling
using Oceananigans
using Oceananigans.Grids: znodes, φnodes
using Oceananigans.Fields: location
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: on_architecture
using XESMF
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using TOML
using JLD2
using Glob
using Statistics

include("./half_degree_omip.jl")

struct ClimaOceanSingleGPUGCPBackend <: ClimaCalibrate.SlurmBackend end

function ClimaCalibrate.module_load_string(::Type{ClimaOceanSingleGPUGCPBackend})
    return """
unset CUDA_HOME CUDA_PATH CUDA_ROOT NVHPC_CUDA_HOME CUDA_INC_DIR CPATH NVHPC_ROOT OPAL_PREFIX
export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v cuda | grep -v ucx | tr '\n' ':' | sed 's/:\$//')
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:\$HOME/cmake-3.28.1-linux-x86_64/bin:\$HOME/julia-1.10.10/bin

export JULIA_CUDA_MEMORY_POOL=binned
export JULIA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

cd \$HOME/CES_oceananigans/ClimaOceanCalibration.jl

# ============================================
# LOAD API KEYS
# ============================================
if [ -f ~/API_keys.sh ]; then
    source ~/API_keys.sh
else
    echo "Warning: API_keys.sh file not found in home directory"
fi

# ============================================
# CONFIGURE FOR SINGLE-GPU (NO UCX)
# ============================================
echo "=== Checking existing configuration ==="
if ~/julia-1.10.10/bin/julia --project=. -e '
using MPI, CUDA
ok = false
rt = try CUDA.runtime_version() catch; nothing end
if rt == v"12.4" && occursin("openmpi", lowercase(MPI.MPI_LIBRARY))
    println("✓ Already configured: OpenMPI + CUDA 12.4")
    exit(0)
else
    exit(1)
end
'; then
    echo "Configuration looks correct; skipping reconfiguration."
else
    echo "=== Configuring for single-GPU ==="
    ~/julia-1.10.10/bin/julia --project=. -e '
    using MPIPreferences
    MPIPreferences.use_jll_binary("OpenMPI_jll")

    using CUDA
    CUDA.set_runtime_version!(v"12.4", local_toolkit=false)

    println("✓ Configured: OpenMPI_jll + CUDA artifacts")
    '
fi

echo "=== Verify Configuration ==="
~/julia-1.10.10/bin/julia --project -e '
using MPI, CUDA, Libdl, Oceananigans, ClimaOcean, ClimaSeaIce

println("MPI: ", MPI.MPI_LIBRARY)
println("CUDA runtime: ", CUDA.runtime_version())

ucx_libs = filter(lib -> occursin("ucx", lowercase(lib)), Libdl.dllist())
if isempty(ucx_libs)
    println("✓ No UCX - safe to run!")
else
    println("⚠️ WARNING: UCX detected:")
    foreach(println, ucx_libs)
    exit(1)
end
'"""
end

backend_worker_kwargs(::Type{ClimaOceanSingleGPUGCPBackend}) = (; partition = "a3mega")

function ClimaCalibrate.forward_model(iteration, member)
    config_dict = Dict()
    
    # Set the output path for the current member
    member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, member)
    config_dict["output_dir"] = member_path

    # Set the parameters for the current member
    parameter_path = ClimaCalibrate.parameter_path(output_dir, iteration, member)
    if haskey(config_dict, "toml")
        push!(config_dict["toml"], parameter_path)
    else
        config_dict["toml"] = [parameter_path]
    end

    params = TOML.parsefile(parameter_path)
    κ_skew = params["κ_skew"]
    κ_symmetric = params["κ_symmetric"]

    try
        run_gm_calibration_omip(κ_skew, κ_symmetric, config_dict)
    catch e
        # Create a failure indicator file with error information
        error_file = joinpath(member_path, "RUN_FAILED.err")
        open(error_file, "w") do io
            println(io, "Run failed at $(now())")
            println(io, "Parameters: κ_skew = $(κ_skew), κ_symmetric = $(κ_symmetric)")
            println(io, "Error: $(e)")
            println(io, "Backtrace:")
            for (exc, bt) in Base.catch_stack()
                showerror(io, exc, bt)
                println(io)
            end
        end
        
        @error "GM calibration failed (κ_skew = $(κ_skew), κ_symmetric = $(κ_symmetric))" exception=(e, catch_backtrace())
    end

    return simulation
end

function regrid_model_data(simdir)
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

    for t in 1:length(times)
        mask_immersed_field!(fts[t], NaN)
    end

    field_section = reshape(z_weights, 1, 1, :, 1) .* interior(fts, :, lat_indices, :, :)

    return field_section
end

extract_southern_ocean_section(fts, vertical_weighting=no_tapering) = extract_field_section(fts, (-80, -50); vertical_weighting)

function process_member_data(simdir)
    T_target, S_target = regrid_model_data(simdir)

    T_section = extract_southern_ocean_section(T_target, taper_interior_ocean)
    S_section = extract_southern_ocean_section(S_target, taper_interior_ocean)
    
    return vcat(vec(T_section), vec(S_section))
end

const ensemble_size = 5
const output_dir = joinpath(pwd(), "calibration_runs", "test_run_gm")

function process_observation(obs_path, vertical_weighting=no_tapering)
    T_filepath = joinpath(obs_path, "T.jld2")
    S_filepath = joinpath(obs_path, "S.jld2")
    
    T_afts = jldopen(T_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    S_afts = jldopen(S_filepath, "r") do file
        return file["averaged_fieldtimeseries"]
    end

    T_data = T_afts.data
    S_data = S_afts.data

    T_section = extract_southern_ocean_section(T_data, vertical_weighting)
    S_section = extract_southern_ocean_section(S_data, vertical_weighting)

    return vcat(T_section[.!isnan.(T_section)], S_section[.!isnan.(S_section)])
end

n_iterations = 3
κ_skew_prior = constrained_gaussian("κ_skew", 5e2, 3e2, 0, Inf)
κ_symmetric_prior = constrained_gaussian("κ_symmetric", 5e2, 3e2, 0, Inf)

priors = combine_distributions([κ_skew_prior, κ_symmetric_prior])

obs_paths = glob("10yearaverage_2degree*", joinpath("calibration_data", "ECCO4Monthly"))
calibration_target_obs_path = obs_paths[findfirst(x -> occursin("2002", x), obs_paths)]

Y = hcat(process_observation.(obs_paths)...)

n_trials = 2
const output_dim = size(Y, 1)

# the noise estimated from the samples (will have rank n_trials-1)
internal_cov = tsvd_cov_from_samples(Y) # SVD object

# the "5%" model error (diagonal)
model_error_frac = 0.05
data_mean = vec(mean(Y,dims=2))
model_error_cov = Diagonal((model_error_frac*data_mean).^2)

# regularize the model error diagonal (in case of zero entries)
model_error_cov += 1e-6*I

# Combine...
covariance = SVDplusD(internal_cov, model_error_cov)

Y_obs = Observation(Dict("samples" => process_observation(calibration_target_obs_path, taper_interior_ocean),
                         "covariances" => covariance,
                         "names" => basename(calibration_target_obs_path)))

utki = EnsembleKalmanProcess(Y_obs, TransformUnscented(priors))

function ClimaCalibrate.observation_map(iteration)
    G_ensemble = zeros(output_dim, ensemble_size)

    for m in 1:ensemble_size
        member_path = ClimaCalibrate.path_to_ensemble_member(output_dir, iteration, m)

        if isfile(joinpath(member_path, "RUN_FAILED.err"))
            @warn "Skipping member $m for iteration $iteration due to failed run."
            G_ensemble[:, m] .= NaN
        else
            G_ensemble[:, m] .= process_member_data(simdir)
        end
    end

    return G_ensemble
end

backend = ClimaOceanSingleGPUGCPBackend

hpc_kwargs = hpc_kwargs = Dict(:ntasks => 1,
                               :cpus_per_task => 4,
                               :gpus_per_task => 1,
                               :mem => "128G")

ClimaCalibrate.calibrate(backend, utki, ensemble_size, n_iterations, priors, output_dir; hpc_kwargs, verbose=true)