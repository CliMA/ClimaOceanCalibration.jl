# precompute_woa_monthly_zonal.jl
# Build the WOA-monthly zonal-mean seasonal target for the seasonal CATKE+GM
# calibration. For each of the 12 WOA climatological months, load WOA monthly
# T/S onto a SHALLOW ORCA grid (the top cells of the forward model's grid,
# truncated to ~1500 m because WOA Monthly only reaches ~1525 m), convert to
# TEOS-10, then run the SAME regrid → zonal-mean → tropics/upper-200 m slicing
# pipeline as the model observation map (data_processing_seasonal.jl). The shallow
# grid's cells coincide with the deep model grid's top cells, so after the
# z ≥ z_min slice the target vector has identical length and ordering to each
# member's (deep-grid) G vector.
#
# Run once before the first seasonal calibration, matching the forward model's
# Δz_top (default `false` ⇒ the omip_simulation default grid, matching
# calibrate_catke_gm_seasonal.jl's `--DZ_TOP false` default):
#   DZ_TOP=false julia +1.12.3 --project=. examples/CATKE_GM_calibration/precompute_woa_monthly_zonal.jl
#   DZ_TOP=1.5   julia +1.12.3 --project=. examples/CATKE_GM_calibration/precompute_woa_monthly_zonal.jl
#
# Output (matches the cache name calibrate_catke_gm_seasonal.jl looks up):
#   Δz_top = nothing ⇒ calibration_data/woa_monthly_zonal.jld2
#   Δz_top = h       ⇒ calibration_data/woa_monthly_zonal_dztop<h>.jld2

using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations: build_grid, upper_orca_grid, woa_to_teos10!
using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAMonthly
using Oceananigans
using Oceananigans.Fields: CenterField, interior
using Dates
using JLD2

include(joinpath(@__DIR__, "data_processing_seasonal.jl"))

const NZ            = 70
const DEPTH         = 5500
const ΔZ_TOP        = let v = lowercase(strip(get(ENV, "DZ_TOP", "false")))
    (v in ("false", "nothing", "default", "")) ? nothing : parse(Float64, v)
end
# WOA Monthly only reaches ~1525 m. Build the target on a SHALLOW grid whose cells
# are the top cells of the full (Nz, depth) model grid, truncated at this depth, so
# WOA fills every level (no out-of-range error / deep extrapolation) while still
# matching a deep-grid member's output cell-for-cell after the z ≥ z_min slice.
const WOA_MAX_DEPTH = parse(Float64, get(ENV, "WOA_MAX_DEPTH", "1500"))
const RESTORING_DIR = joinpath(homedir(), "ECCO_data")

output_dir = joinpath(@__DIR__, "calibration_data")
mkpath(output_dir)
output_file = joinpath(output_dir,
    ΔZ_TOP === nothing ? "woa_monthly_zonal.jld2" : "woa_monthly_zonal_dztop$(ΔZ_TOP).jld2")

mkpath(RESTORING_DIR)

@info "Building SHALLOW ORCA grid (top cells of Nz=$NZ, depth=$DEPTH to ~$(WOA_MAX_DEPTH)m, Δz_top=$ΔZ_TOP)..."
grid = upper_orca_grid(CPU(), NZ, DEPTH, WOA_MAX_DEPTH; Δz_top = ΔZ_TOP)
@info "  shallow grid has $(size(grid, 3)) levels (bottom ≈ $(round(minimum(znodes(grid, Center())); digits = 1)) m)"

@info "Loading 12 WOA monthly T/S slices onto the shallow ORCA grid + TEOS-10 conversion..."
# `set!` works here because the shallow grid (≈$(WOA_MAX_DEPTH)m) is within WOA
# Monthly's vertical range (~1525 m): every level is filled, none extrapolated.
Ts = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
Ss = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
Bs = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
for m in 1:SEASONAL_N_MONTHS
    date = DateTime(2018, m, 1)
    T = CenterField(grid)
    S = CenterField(grid)
    set!(T, Metadatum(:temperature; dir = RESTORING_DIR, dataset = WOAMonthly(), date = date))
    set!(S, Metadatum(:salinity;    dir = RESTORING_DIR, dataset = WOAMonthly(), date = date))
    woa_to_teos10!(T, S)
    Ts[m] = Array(interior(T))
    Ss[m] = Array(interior(S))
    Bs[m] = seasonal_buoyancy_3d(T, S)
    @info "  month $m done"
end

@info "Regridding + zonal-averaging WOA monthly T,S,b (single regridder)..."
rg       = seasonal_regridder(grid)
mask     = seasonal_ocean_mask_3d(grid)
latitude = seasonal_zonal_latitudes()
depth    = collect(znodes(grid, Center()))
zonal_T  = [seasonal_zonal_mean(Ts[m], mask, rg) for m in 1:SEASONAL_N_MONTHS]
zonal_S  = [seasonal_zonal_mean(Ss[m], mask, rg) for m in 1:SEASONAL_N_MONTHS]
zonal_b  = [seasonal_zonal_mean(Bs[m], mask, rg) for m in 1:SEASONAL_N_MONTHS]
Y_target = flatten_seasonal_zonal(zonal_T, zonal_S, latitude, depth;
                                  lat_range = SEASONAL_LAT_RANGE, z_min = SEASONAL_Z_MIN)

@info "Saving WOA monthly zonal target to $output_file (length $(length(Y_target)))"
jldopen(output_file, "w") do file
    file["Y_target"]   = Y_target
    # 2-D zonal arrays + axes, for the per-member seasonal video (WOA reference).
    file["zonal_T"]    = zonal_T          # 12-vector of (Nlat, Nz)
    file["zonal_S"]    = zonal_S
    file["zonal_b"]    = zonal_b
    file["latitude"]   = latitude         # lat-lon latitude axis (Nlat)
    file["depth"]      = depth            # model depth axis (Nz)
    file["Nz"]         = NZ
    file["model_depth"] = DEPTH
    file["max_depth"]  = WOA_MAX_DEPTH
    file["Δz_top"]     = ΔZ_TOP
    file["lat_range"]  = SEASONAL_LAT_RANGE
    file["z_min"]      = SEASONAL_Z_MIN
    file["n_months"]   = SEASONAL_N_MONTHS
end

@info "Done. Seasonal target length = $(length(Y_target)) (= 12 months × (T + S) tropical/upper-200 m cells)."
