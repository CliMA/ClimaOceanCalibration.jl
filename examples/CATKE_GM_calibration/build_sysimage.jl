# build_sysimage.jl
#
# Builds a PackageCompiler sysimage that bakes in the compiled code for the
# CATKE+GM ORCA calibration forward model, so each cold per-member Julia
# process started by the calibration (one per ensemble member, hundreds per
# campaign) skips re-compiling the Oceananigans / NumericalEarth / CUDA stack.
#
# Run under the root project (PackageCompiler is a root dep), on an a3mega H100
# node (the trace runs a real GPU forward model):
#
#   julia +1.12.3 --project=. examples/CATKE_GM_calibration/build_sysimage.jl
#
# The output path is taken from CALIBRATION_SYSIMAGE when set (the launch
# scripts point it at a Manifest-hash-keyed file so a sysimage is never reused
# across an incompatible Manifest). Otherwise it defaults to
# <repo>/sysimages/catke_gm_orca.so.
#
# CAVEATS:
#   - The sysimage is tied to the *current* Manifest.toml and to the H100 / CUDA
#     toolchain. Rebuild after any Pkg update; build on the same node image used
#     to run calibration. The launch scripts key the filename on the Manifest
#     hash, so a Manifest change triggers an automatic rebuild.
#   - A functional GPU is required at build time (the trace uses arch = GPU()).
#   - Build cost ≈ one member's compile + 2 time steps, paid once per Manifest.

using PackageCompiler

project  = abspath(joinpath(@__DIR__, "..", ".."))   # repo root Project.toml
sysimage = get(ENV, "CALIBRATION_SYSIMAGE",
               joinpath(project, "sysimages", "catke_gm_orca.so"))
mkpath(dirname(sysimage))

@info "Building sysimage" project sysimage

create_sysimage(
    ["ClimaOceanCalibration", "Oceananigans", "ClimaCalibrate",
     "EnsembleKalmanProcesses", "CUDA", "JLD2"];
    sysimage_path             = sysimage,
    project                   = project,
    precompile_execution_file = joinpath(@__DIR__, "precompile_workload.jl"),
)

@info "Sysimage build complete" sysimage
