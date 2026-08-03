# postprocess_flux_climatology.jl
#
# Convert the raw monthly output of prescribed_woa_fluxes.jl into CMIP-named,
# CMIP-signed monthly climatologies, in the same NetCDF layout produced by
# fetch_omip2_fluxes.py for the OMIP2 models — so plot_flux_comparison.jl can
# treat our run exactly like another model.
#
# Usage:
#   julia +1.12.3 --project=<repo> postprocess_flux_climatology.jl <run>_monthly_fluxes.jld2 [out_dir]
#
# For each derived variable it writes  <out_dir>/<var>_omip2_<LABEL>.nc  with
#   clim   (12, Nx→x, Ny→y … stored as (month, y, x)) native ORCA climatology
#   lat/lon (y, x) native coordinates
#   clim_r (12, 180, 360) 1° lat-lon bin-average regrid
#   lat_r/lon_r 1° axes
# LABEL = "NumericalEarth-<flux_config>".
#
# Derived variables and conventions (CMIP):
#   hfls, hfss   W m⁻², positive UP     (raw fields already positive-up)
#   evs          kg m⁻² s⁻¹, positive UP
#   prra, prsn   kg m⁻² s⁻¹, positive DOWN
#   friver       kg m⁻² s⁻¹ (runoff + calving; JRA55-do)
#   wfo          kg m⁻² s⁻¹, positive INTO ocean = prra + prsn + friver − evs
#   hfds         W m⁻², positive DOWN   = −ρ₀ cₚ · JT   (JT: Oceananigans BC sign, + = cooling)
#   rsntds       W m⁻², positive DOWN   (net/transmitted shortwave)
#   rlntds       W m⁻², positive DOWN   (absorbed − emitted longwave)
#   tauuo,tauvo  N m⁻², downward flux of eastward/northward momentum = −ρτ(raw)
#
# Signs that depend on internal storage conventions (radiative fields, stress,
# JT) are VERIFIED at runtime with region-mean assertions and flipped loudly
# if needed — see `oriented`.

using JLD2
using NCDatasets
using Printf
using Statistics

include(joinpath(@__DIR__, "binavg_regrid.jl"))

# ============================================
# Input
# ============================================
length(ARGS) ≥ 1 || error("usage: julia postprocess_flux_climatology.jl <run>_monthly_fluxes.jld2 [out_dir]")
const in_path = abspath(ARGS[1])
isfile(in_path) || error("input not found: $in_path")

raw   = jldopen(in_path, "r")
meta  = Dict(k => raw["metadata/$k"] for k in keys(raw["metadata"]))
lon   = raw["grid/lon"]  :: Matrix
lat   = raw["grid/lat"]  :: Matrix
wet   = raw["grid/wet"]  :: Matrix{Bool}

const flux_config = meta["flux_config"]
const label       = "NumericalEarth-$(flux_config)"
const ρ₀ = get(meta, "reference_density", 1025.0)
const cₚ = get(meta, "heat_capacity", 4000.0)

const out_dir = length(ARGS) ≥ 2 ? abspath(ARGS[2]) : joinpath(dirname(in_path), "climatology")
mkpath(out_dir)

month_keys = sort(parse.(Int, keys(raw["monthly/time"])))
nmonths    = length(month_keys)
navailable = nmonths ÷ 12
navailable ≥ 1 || error("need at least 12 complete months, found $nmonths")
# CLIM_YEARS: how many trailing years enter the climatology (default: all
# complete years). CLIM_YEARS=1 mirrors the seasonal calibration's
# last-year-of-run target convention.
nyears = clamp(parse(Int, get(ENV, "CLIM_YEARS", string(navailable))), 1, navailable)
used = month_keys[end-12nyears+1:end]   # last `nyears` complete years
@info "Input: $nmonths months; climatology over the last $nyears years (months $(first(used))–$(last(used)))"

# 12-month climatology of a raw monthly series
function clim_of(name)
    Nx, Ny = size(lon)
    c = zeros(Float64, 12, Nx, Ny)
    n = zeros(Int, 12)
    for m in used
        mm = mod1(m, 12)
        c[mm, :, :] .+= Float64.(raw["monthly/$name/$m"])
        n[mm] += 1
    end
    for mm in 1:12
        c[mm, :, :] ./= n[mm]
    end
    # mask land
    for mm in 1:12
        cm = view(c, mm, :, :)
        cm[.!wet] .= NaN
    end
    return c
end

# ============================================
# Assemble CMIP-convention variables
# ============================================
# All storage-sign conversions below are FIXED, per the coupled integration
# test documented in README.md §"Verified conventions" (2026-07): turbulent
# fluxes are stored +up, radiative diagnostics +down (down) / +up (rlus),
# ρτ is stored as the upward flux of momentum (negative under a westerly),
# and net fluxes carry the Oceananigans BC sign (positive = out of the ocean).
# `sanity` only CHECKS region means and warns — it never flips data, so a
# violated check means either upstream conventions changed or the run is bad.
tropics = @. abs(lat) < 15
midlatN = @. (lat > 35) & (lat < 55)

region_mean(c, region) = begin
    sel = region .& wet
    mean(filter(isfinite, [mean(c[mm, i, j] for mm in 1:12) for (i, j) in Tuple.(findall(sel))]))
end

nwarn = Ref(0)
function sanity(name, c, region, region_name, should_be_positive)
    μ = region_mean(c, region)
    ok = should_be_positive ? μ > 0 : μ < 0
    if !ok
        nwarn[] += 1
        @warn "$name: $region_name mean $(round(μ, digits=2)) violates the expected sign " *
              "($(should_be_positive ? "+" : "−")). NOT flipping — check whether upstream " *
              "storage conventions changed (see README §Verified conventions)."
    else
        @info @sprintf("%-7s %s mean = %+.2f ✓", name, region_name, μ)
    end
    return c
end

vars = Dict{String, Array{Float64, 3}}()

vars["hfls"] = sanity("hfls", clim_of("hfls"), tropics, "tropical", true)   # + up
vars["hfss"] = clim_of("hfss")                                              # + up
vars["evs"]  = sanity("evs",  clim_of("evs"),  tropics, "tropical", true)   # + up
vars["prra"] = sanity("prra", clim_of("prra"), tropics, "tropical", true)   # + down
vars["prsn"] = clim_of("prsn")
vars["friver"] = clim_of("friver")

# net heat into ocean (+down, warming): JT is +up (cooling) in K m/s
vars["hfds"] = sanity("hfds", -ρ₀ * cₚ .* clim_of("JT"), tropics, "tropical", true)

# radiation (verified storage: rsds/rlds +down, rlus +up)
vars["rsntds"] = sanity("rsntds", clim_of("rsds"), tropics, "tropical", true)
vars["rlntds"] = sanity("rlntds", clim_of("rlds") .- clim_of("rlus"), tropics, "tropical", false)

# net freshwater into ocean
vars["wfo"] = vars["prra"] .+ vars["prsn"] .+ vars["friver"] .- vars["evs"]

# wind stress on ocean (+ = downward flux of eastward momentum): ρτ is stored
# as the upward momentum flux, so CMIP tauuo/tauvo = −ρτ (verified).
vars["tauuo"] = sanity("tauuo", .-clim_of("rtauuo"), midlatN, "N-midlat", true)
vars["tauvo"] = .-clim_of("rtauvo")

# context fields
vars["tos"]  = clim_of("tos") .- 273.15   # K → °C for comparison with tos maps
vars["ustar"] = clim_of("ustar")

# closure check: hfds = rsntds + rlntds − hfls − hfss must hold exactly (no ice)
resid = vars["hfds"] .- (vars["rsntds"] .+ vars["rlntds"] .- vars["hfls"] .- vars["hfss"])
resid_mean = region_mean(resid, trues(size(lat)))
resid_max  = maximum(abs, filter(isfinite, resid))
if resid_max > 1
    nwarn[] += 1
    @warn @sprintf("heat closure VIOLATED: residual mean %.2f, max |%.2f| W/m² — output signs are suspect!",
                   resid_mean, resid_max)
else
    @info @sprintf("heat closure OK (residual mean %.2e, max %.2e W/m²)", resid_mean, resid_max)
end
nwarn[] == 0 || @warn "$(nwarn[]) sanity check(s) failed — inspect before using the output."

# ============================================
# 1° bin-average regrid + NetCDF output (shared with fetch_omip2_fluxes.jl,
# see binavg_regrid.jl — same operator on both sides of the comparison)
# ============================================
for (var, c) in sort(collect(vars); by = first)
    out_path = joinpath(out_dir, "$(var)_omip2_$(label).nc")
    clim_r = regrid_binavg_1deg(c, lon, lat, wet)
    write_climatology_netcdf(out_path, var, label, "prescribed-WOA23", c, lon, lat, clim_r;
                             clim_months = 12 * nyears,
                             note = "JRA55-do fluxes over prescribed WOA23 monthly SST/SSS " *
                                    "(no dynamic ocean, no sea ice); climatology of last " *
                                    "$nyears years starting $(meta["start_year"])",
                             extra_attrs = ["flux_config" => flux_config])
    @info "wrote $(basename(out_path))"
end

close(raw)
@info "Done. Climatologies in $out_dir — copy/symlink into omip_data/omip2_fluxes/ for plot_flux_comparison.jl."
