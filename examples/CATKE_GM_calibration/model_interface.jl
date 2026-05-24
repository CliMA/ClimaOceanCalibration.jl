# model_interface.jl
#
# Thin adapter shimming the existing 2-arg ClimaCalibrate hooks in
# model_interface_v01.jl onto the v0.3.0 AbstractModelInterface (3-arg) API.
#
# The legacy model_interface_v01.jl requires `output_dir` to be defined as a
# global *before* it is included. Both the parent driver
# (calibrate_catke_gm.jl) and the child forward-model script (see
# batched_job_body in batched_slurm_backend.jl) set `output_dir`
# before including this file, so the legacy file's @isdefined check passes
# in both contexts.

using ClimaCalibrate

# Bring in the legacy 2-arg methods:
#   ClimaCalibrate.forward_model(iter, member)
#   ClimaCalibrate.observation_map(iter)
#   ClimaCalibrate.analyze_iteration(ekp, g, prior, output_dir, iter)
include(joinpath(@__DIR__, "model_interface_v01.jl"))

struct CATKEGMInterface <: ClimaCalibrate.AbstractModelInterface
    model_interface_path::String
    project_path::String
end

function CATKEGMInterface(;
    model_interface_path = abspath(@__FILE__),
    project_path         = abspath(joinpath(@__DIR__, "..", "..")),
)
    return CATKEGMInterface(model_interface_path, project_path)
end

# Dispatch the v0.3.0 3-arg interface to the legacy 2-arg methods that the
# include above already attached to ClimaCalibrate.forward_model etc.
ClimaCalibrate.forward_model(::CATKEGMInterface, iter, member) =
    ClimaCalibrate.forward_model(iter, member)

ClimaCalibrate.observation_map(::CATKEGMInterface, iter) =
    ClimaCalibrate.observation_map(iter)

ClimaCalibrate.analyze_iteration(
    ::CATKEGMInterface, ekp, g_ensemble, prior, out_dir, iter,
) = ClimaCalibrate.analyze_iteration(ekp, g_ensemble, prior, out_dir, iter)

ClimaCalibrate.model_interface_filepath(i::CATKEGMInterface) = i.model_interface_path
ClimaCalibrate.experiment_dir(i::CATKEGMInterface)            = i.project_path
