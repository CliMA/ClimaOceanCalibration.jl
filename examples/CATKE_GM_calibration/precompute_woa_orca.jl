# precompute_woa_orca.jl
# Build the ORCA grid the same way omip_simulation(:orca; ...) does, load WOA
# annual T/S onto it, convert to TEOS-10 (Conservative T, Absolute S) the same
# way build_ocean does, and save the result for use as the calibration target.
#
# Run once before the first calibration:
#   julia +1.12.3 --project=. examples/CATKE_GM_calibration/precompute_woa_orca.jl
#
# Output: examples/CATKE_GM_calibration/calibration_data/woa_orca_grid.jld2

using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations: build_grid, woa_to_teos10!
using NumericalEarth
using NumericalEarth.DataWrangling
using Oceananigans
using Oceananigans.Fields: CenterField, interior
using Oceananigans.Architectures: on_architecture
using JLD2

const NZ            = 70
const DEPTH         = 5500
const RESTORING_DIR = get(ENV, "RESTORING_DIR", joinpath(homedir(), "ECCO_data"))

output_dir = joinpath(@__DIR__, "calibration_data")
mkpath(output_dir)
output_file = joinpath(output_dir, "woa_orca_grid.jld2")

mkpath(RESTORING_DIR)

@info "Building ORCA grid (Nz=$NZ, depth=$DEPTH)..."
grid = build_grid(Val(:orca), CPU(), NZ, DEPTH)

@info "Loading WOA annual T/S and interpolating onto the ORCA grid..."
T = CenterField(grid)
S = CenterField(grid)
set!(T, Metadatum(:temperature; dir=RESTORING_DIR, dataset=WOAAnnual()))
set!(S, Metadatum(:salinity;    dir=RESTORING_DIR, dataset=WOAAnnual()))

@info "Converting WOA in-situ T / Practical S to TEOS-10 Conservative T / Absolute S..."
woa_to_teos10!(T, S)

T_arr = Array(interior(T))
S_arr = Array(interior(S))

@info "Saving WOA-on-ORCA to $output_file ($(size(T_arr)))"
jldopen(output_file, "w") do file
    file["T"]     = T_arr
    file["S"]     = S_arr
    file["Nz"]    = NZ
    file["depth"] = DEPTH
end

@info "Done."
