# One-degree OMIP simulation of ClimaOcean's `OneDegree()` configuration with the maximum isopycnal slope of the
# Gent-McWilliams and Redi flux tapering set to `max_slope`.

using ArgParse
using ClimaOcean
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, FluxTapering
using CUDA

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--max-slope"
        arg_type = Float64
        default = 1e-2
    "--years"
        arg_type = Int
        default = 20
    "--output-directory"
        arg_type = String
        default = "gm_tapering"
    "--forcing-directory"
        arg_type = String
        default = expanduser("~/JRA55_data")
    "--pickup"
        help = "checkpoint file to restart from"
        arg_type = String
        default = ""
end
arguments = parse_args(settings)

max_slope = arguments["max-slope"]
years = arguments["years"]
forcing_directory = arguments["forcing-directory"]
name = "taper_" * string(max_slope)
output_directory = joinpath(arguments["output-directory"], name)

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = 500, κ_symmetric = 800,
                                                 slope_limiter = FluxTapering(max_slope))

configuration = OneDegree(; name, eddy_closure)
@info configuration

simulation = omip_simulation(configuration; arch = GPU(), forcing_directory, output_directory,
                             stop_time = years * 365days)

pickup = isempty(arguments["pickup"]) ? false : arguments["pickup"]
run!(simulation; pickup)
