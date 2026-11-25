# precompute_woa_orca.jl
# Load WOA annual T/S onto the `OneDegree()` grid of the forward model, convert
# to TEOS-10 (Conservative T, Absolute S), and save the result for use as the
# calibration target. Run once before the first calibration:
#   julia +1.13.3 --project=. examples/CATKE_GM_calibration/precompute_woa_orca.jl
#
# Output (the cache name calibrate_catke_gm.jl looks up):
#   calibration_data/woa_one_degree.jld2

using ClimaOcean: OneDegree, omip_grid
using ClimaOceanCalibration.OMIPSimulations: woa_to_teos10!
using NumericalEarth
using NumericalEarth.DataWrangling
using Oceananigans
using Oceananigans.Fields: CenterField, interior
using JLD2


output_dir = joinpath(@__DIR__, "calibration_data")
mkpath(output_dir)
output_file = joinpath(output_dir, "woa_one_degree.jld2")

@info "Building the OneDegree() grid..."
grid = omip_grid(OneDegree(), CPU())

@info "Loading WOA annual T/S and interpolating onto the ORCA grid..."
T = CenterField(grid)
S = CenterField(grid)
set!(T, Metadatum(:temperature; dataset=WOAAnnual()))
set!(S, Metadatum(:salinity;    dataset=WOAAnnual()))

@info "Converting WOA in-situ T / Practical S to TEOS-10 Conservative T / Absolute S..."
woa_to_teos10!(T, S)

T_arr = Array(interior(T))
S_arr = Array(interior(S))

@info "Saving WOA-on-ORCA to $output_file ($(size(T_arr)))"
jldopen(output_file, "w") do file
    file["T"] = T_arr
    file["S"] = S_arr
end

@info "Done."
