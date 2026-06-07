# precompute_woa_orca.jl
# Build the ORCA grid the same way omip_simulation(:orca; ...) does, load WOA
# annual T/S onto it, convert to TEOS-10 (Conservative T, Absolute S) the same
# way build_ocean does, and save the result for use as the calibration target.
#
# Run once before the first calibration, matching the forward model's Δz_top. The
# surface-cell thickness is set via the DZ_TOP env var (default 1.5); `false` or
# `nothing` uses the omip_simulation default grid:
#   DZ_TOP=1.5   julia +1.12.3 --project=. examples/CATKE_GM_calibration/precompute_woa_orca.jl
#   DZ_TOP=false julia +1.12.3 --project=. examples/CATKE_GM_calibration/precompute_woa_orca.jl
#
# Output (matches the cache name calibrate_catke_gm.jl looks up):
#   Δz_top = nothing ⇒ calibration_data/woa_orca_grid.jld2
#   Δz_top = h       ⇒ calibration_data/woa_orca_grid_dztop<h>.jld2

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
# DZ_TOP=false (or nothing/default/unset) ⇒ default grid; otherwise the surface-cell thickness (m).
const ΔZ_TOP        = let v = lowercase(strip(get(ENV, "DZ_TOP", "1.5")))
    (v in ("false", "nothing", "default", "")) ? nothing : parse(Float64, v)
end
const RESTORING_DIR = joinpath(homedir(), "ECCO_data")

output_dir = joinpath(@__DIR__, "calibration_data")
mkpath(output_dir)
output_file = joinpath(output_dir,
    ΔZ_TOP === nothing ? "woa_orca_grid.jld2" : "woa_orca_grid_dztop$(ΔZ_TOP).jld2")

mkpath(RESTORING_DIR)

@info "Building ORCA grid (Nz=$NZ, depth=$DEPTH, Δz_top=$ΔZ_TOP)..."
grid = build_grid(Val(:orca), CPU(), NZ, DEPTH; Δz_top = ΔZ_TOP)

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
    file["T"]      = T_arr
    file["S"]      = S_arr
    file["Nz"]     = NZ
    file["depth"]  = DEPTH
    file["Δz_top"] = ΔZ_TOP
end

@info "Done."
