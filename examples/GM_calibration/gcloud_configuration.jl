using ClimaCalibrate
using ClimaCalibrate: generate_sbatch_directives
import ClimaCalibrate: generate_sbatch_script

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

ClimaCalibrate.backend_worker_kwargs(::Type{ClimaOceanSingleGPUGCPBackend}) = (; partition = "a3mega")

function generate_sbatch_script(iter::Int, member::Int, output_dir, experiment_dir, model_interface, module_load_str, hpc_kwargs, exeflags = "")
    member_log = path_to_model_log(output_dir, iter, member)
    slurm_directives = generate_sbatch_directives(hpc_kwargs)

    sbatch_contents = """
    #!/bin/bash
    #SBATCH --job-name=run_$(iter)_$(member)
    #SBATCH --output=$member_log
    $slurm_directives

    $module_load_str

    julia $exeflags --project=$experiment_dir -e '

        import ClimaCalibrate as CAL
        iteration = $iter; member = $member
        model_interface = "$model_interface"; include(model_interface)
        experiment_dir = "$experiment_dir"
        CAL.forward_model(iteration, member)
        CAL.write_model_completed("$output_dir", iteration, member)
    '
    exit 0
    """
    return sbatch_contents
end