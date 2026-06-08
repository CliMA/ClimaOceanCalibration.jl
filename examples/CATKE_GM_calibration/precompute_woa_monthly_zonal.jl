# precompute_woa_monthly_zonal.jl
# Build the WOA-monthly zonal-mean seasonal target for the seasonal CATKE+GM
# calibration. For each of the 12 WOA climatological months, load WOA monthly
# T/S onto the ORCA grid (matching the forward model's grid), convert to TEOS-10,
# then run the SAME regrid → zonal-mean → tropics/upper-200 m slicing pipeline as
# the model observation map (data_processing_seasonal.jl). This guarantees the
# target vector has identical length and ordering to each member's G vector.
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
using ClimaOceanCalibration.OMIPSimulations: build_grid, woa_to_teos10!
using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAMonthly
using Oceananigans
using Oceananigans.Fields: Field, CenterField, interior, interpolate!
using Dates
using JLD2

include(joinpath(@__DIR__, "data_processing_seasonal.jl"))

const NZ            = 70
const DEPTH         = 5500
const ΔZ_TOP        = let v = lowercase(strip(get(ENV, "DZ_TOP", "false")))
    (v in ("false", "nothing", "default", "")) ? nothing : parse(Float64, v)
end
const RESTORING_DIR = joinpath(homedir(), "ECCO_data")

output_dir = joinpath(@__DIR__, "calibration_data")
mkpath(output_dir)
output_file = joinpath(output_dir,
    ΔZ_TOP === nothing ? "woa_monthly_zonal.jld2" : "woa_monthly_zonal_dztop$(ΔZ_TOP).jld2")

mkpath(RESTORING_DIR)

@info "Building ORCA grid (Nz=$NZ, depth=$DEPTH, Δz_top=$ΔZ_TOP)..."
grid = build_grid(Val(:orca), CPU(), NZ, DEPTH; Δz_top = ΔZ_TOP)

@info "Loading 12 WOA monthly T/S slices onto the ORCA grid + TEOS-10 conversion..."
# NOTE: WOA Monthly only reaches ~1525 m, so `set!(field_on_5500m_grid, Metadatum)`
# errors ("vertical range ... smaller than the target grid"). Instead build the
# field on WOA's native grid and `interpolate!` onto the (deep) model grid — the
# same pattern the visualization cache uses. The model grid is kept (not a shallow
# grid) so the target matches each member's observation map cell-for-cell; the
# levels below WOA's range are extrapolated but discarded by the z ≥ z_min slice
# (we only calibrate/visualize the upper ocean).
Ts = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
Ss = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
Bs = Vector{Array{Float64, 3}}(undef, SEASONAL_N_MONTHS)
for m in 1:SEASONAL_N_MONTHS
    date = DateTime(2018, m, 1)
    woaT = Field(Metadatum(:temperature; dir = RESTORING_DIR, dataset = WOAMonthly(), date = date), CPU())
    woaS = Field(Metadatum(:salinity;    dir = RESTORING_DIR, dataset = WOAMonthly(), date = date), CPU())
    T = CenterField(grid)
    S = CenterField(grid)
    interpolate!(T, woaT)
    interpolate!(S, woaS)
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
    file["Δz_top"]     = ΔZ_TOP
    file["lat_range"]  = SEASONAL_LAT_RANGE
    file["z_min"]      = SEASONAL_Z_MIN
    file["n_months"]   = SEASONAL_N_MONTHS
end

@info "Done. Seasonal target length = $(length(Y_target)) (= 12 months × (T + S) tropical/upper-200 m cells)."
