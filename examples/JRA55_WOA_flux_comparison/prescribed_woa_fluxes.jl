# prescribed_woa_fluxes.jl
#
# Air–sea fluxes from the JRA55(-do) atmosphere over a PRESCRIBED ocean surface
# given by the WOA23 monthly climatology, on the ORCA (1°, NEMO eORCA1) surface
# grid. No dynamic ocean is run: the ocean component is NumericalEarth's
# `PrescribedOcean`, whose SST/SSS are updated every iteration from a cyclic
# 12-month WOA climatology. The atmosphere, downwelling radiation and land
# runoff are the same prescribed JRA55 components as production OMIP runs
# (src/OMIPSimulations/atmosphere.jl); the bulk-flux formulation is selected
# with FLUX_CONFIG (corrected = COARE 3.6 production physics, ncar = Large &
# Yeager OMIP-2 protocol formulae) and reuses the exact constructors from
# src/OMIPSimulations/omip_simulation.jl.
#
# Monthly-mean flux maps are accumulated online and written to
#   <OUTPUT_DIR>/<RUN_NAME>_monthly_fluxes.jld2
# Post-process with postprocess_flux_climatology.jl, compare with
# plot_flux_comparison.jl.
#
# Environment-variable configuration (defaults in parentheses):
#   FLUX_CONFIG   corrected | ncar                     (corrected)
#   START_YEAR    first JRA55 year                     (1958 — same start as the
#                 calibrate_catke_gm_seasonal.jl forward runs / omip_simulation)
#   STOP_YEARS    run length in years (may be fractional)  (5, matching the
#                 calibration's simulation_length)
#   REPEAT_YEAR   "true" → RepeatYearJRA55 (single repeating year; small-download
#                 demo mode). Default "false" → MultiYearJRA55.
#   DT_HOURS      coupling time step in hours          (1.0; must divide 730)
#   ARCH          gpu | cpu                            (gpu)
#   FORCING_DIR   JRA55 data directory                 (~/JRA55_data)
#   RESTORING_DIR WOA/climatology data directory       (~/ECCO_data)
#   OUTPUT_DIR    output directory                     (<RUN_NAME>_run)
#   RUN_NAME      run name                             (auto from config)
#   BACKEND_SIZE  JRA55 time indices in memory         (240)
#   NZ, DEPTH     production grid the surface cell is taken from (70, 5500)
#   SURFACE_DEPTH keep grid cells above this depth [m] (6.0 → single top cell)
#
# Run (HPC): ./launch_fluxes.sh   or directly:
#   julia +1.12.3 --project=<ClimaOceanCalibration.jl> -t 4 prescribed_woa_fluxes.jl

# DataDeps consent for ORCA/WOA/JRA55 auto-downloads (no-op when already present)
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# OMIPSimulations is self-contained (only NumericalEarth/Oceananigans deps), so
# include it directly rather than via `using ClimaOceanCalibration`, which pulls
# in plotting and calibration dependencies that are irrelevant here.
include(joinpath(@__DIR__, "..", "..", "src", "OMIPSimulations", "OMIPSimulations.jl"))
using .OMIPSimulations: upper_orca_grid,
                        corrected_atmosphere_ocean_fluxes,
                        ncar_atmosphere_ocean_fluxes
using NumericalEarth
using NumericalEarth.Oceans: PrescribedOcean
using NumericalEarth.EarthSystemModels: AtmosphereOceanModel
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentInterfaces,
                                                              RelativeVelocity
using NumericalEarth.Radiations: SurfaceRadiationProperties
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAMonthly
using NumericalEarth.DataWrangling.JRA55: MultiYearJRA55, RepeatYearJRA55,
                                          JRA55PrescribedAtmosphere,
                                          JRA55PrescribedRadiation,
                                          JRA55PrescribedLand
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: CenterField, interior
using Oceananigans.Grids: λnodes, φnodes, inactive_cell, on_architecture
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical
using Oceananigans.Units: Time
using Oceananigans.Utils: prettytime
using CUDA
using Dates
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "prescribed_ocean_patches.jl"))

# ============================================
# Configuration
# ============================================
const flux_config   = lowercase(get(ENV, "FLUX_CONFIG", "corrected"))
const start_year    = parse(Int, get(ENV, "START_YEAR", "1958"))
const stop_years    = parse(Float64, get(ENV, "STOP_YEARS", "5"))
const repeat_year   = lowercase(get(ENV, "REPEAT_YEAR", "false")) == "true"
const Δt_hours      = parse(Float64, get(ENV, "DT_HOURS", "1.0"))
const arch_str      = lowercase(get(ENV, "ARCH", "gpu"))
const forcing_dir   = get(ENV, "FORCING_DIR", joinpath(homedir(), "JRA55_data"))
const restoring_dir = get(ENV, "RESTORING_DIR", joinpath(homedir(), "ECCO_data"))
const backend_size  = parse(Int, get(ENV, "BACKEND_SIZE", "240"))
const Nz_production = parse(Int, get(ENV, "NZ", "70"))
const depth         = parse(Float64, get(ENV, "DEPTH", "5500"))
const surface_depth = parse(Float64, get(ENV, "SURFACE_DEPTH", "6.0"))

flux_config in ("corrected", "ncar") ||
    error("FLUX_CONFIG must be 'corrected' or 'ncar', got '$flux_config'")

const stop_label = isinteger(stop_years) ? string(Int(stop_years)) : string(stop_years)
const run_name   = get(ENV, "RUN_NAME",
                       "woafluxes_$(flux_config)_$(start_year)_$(stop_label)yr" *
                       (repeat_year ? "_repeatyr" : ""))
const output_dir = get(ENV, "OUTPUT_DIR", "$(run_name)_run")

# One climatological month; DT_HOURS must divide it (365/12 day = 730 h).
const MONTH = 365days / 12
const Δt    = Δt_hours * 1hours
@assert isinteger(MONTH / Δt) "DT_HOURS=$Δt_hours must divide the 730 h climatological month"

const start_date = DateTime(start_year, 1, 1)
const end_date   = min(start_date + Year(ceil(Int, stop_years) + 1), DateTime(2019, 1, 1))
const stop_time  = stop_years * 365days

arch = arch_str == "gpu" ? GPU() : CPU()

mkpath(output_dir)
mkpath(forcing_dir)
mkpath(restoring_dir)

@info """JRA55 × WOA prescribed-ocean flux run
    flux_config   = $flux_config
    window        = $start_date … $(start_date + Year(stop_years)) ($stop_years yr)
    Δt            = $(prettytime(Δt))
    arch          = $arch
    forcing_dir   = $forcing_dir
    restoring_dir = $restoring_dir
    output        = $(joinpath(output_dir, run_name * "_monthly_fluxes.jld2"))
"""

# ============================================
# Grid: top cell(s) of the production ORCA grid
# ============================================
grid = upper_orca_grid(arch, Nz_production, depth, surface_depth)
Nx, Ny, Nzs = size(grid)
@info "Surface ORCA grid: $(Nx)×$(Ny)×$(Nzs) (cells above $(surface_depth) m of the Nz=$Nz_production, depth=$depth m production grid)"

# ============================================
# WOA23 monthly SST/SSS slabs (cyclic FieldTimeSeries)
# ============================================
# Same load path as examples/CATKE_GM_calibration/precompute_woa_monthly_zonal.jl.
# PrescribedOcean expects temperature in KELVIN (temperature_units = DegreesKelvin);
# WOA t_an is in-situ °C. At the surface in-situ ≈ potential ≈ conservative T, so
# no TEOS-10 conversion is applied (see PLAN.md §5.5). Salinity: practical, psu.
const celsius_to_kelvin = 273.15
const T_fill_K = 271.35        # fill for active cells WOA misses (freezing seawater)
const S_fill   = 35.0

woa_times = [(m - 0.5) * MONTH for m in 1:12]
T_woa = FieldTimeSeries{Center, Center, Nothing}(grid, woa_times; time_indexing = Cyclical(365days))
S_woa = FieldTimeSeries{Center, Center, Nothing}(grid, woa_times; time_indexing = Cyclical(365days))

@info "Loading 12 WOA23 monthly T/S slabs onto the surface grid..."
nfillT = 0
for m in 1:12
    T3 = CenterField(grid)
    S3 = CenterField(grid)
    set!(T3, Metadatum(:temperature; dir = restoring_dir, dataset = WOAMonthly(), date = DateTime(2018, m, 1)))
    set!(S3, Metadatum(:salinity;    dir = restoring_dir, dataset = WOAMonthly(), date = DateTime(2018, m, 1)))

    Th = Array(interior(T3, :, :, Nzs))
    Sh = Array(interior(S3, :, :, Nzs))
    badT = isnan.(Th); badS = isnan.(Sh)
    global nfillT += count(badT)
    Th[badT] .= T_fill_K - celsius_to_kelvin
    Sh[badS] .= S_fill
    Th .+= celsius_to_kelvin

    copyto!(interior(T_woa[m], :, :, 1), Th)
    copyto!(interior(S_woa[m], :, :, 1), Sh)
end
nfillT > 0 && @warn "Filled $nfillT NaN SST cells (over 12 months) with $(T_fill_K) K — masked in postprocessing via the wet mask."

# ============================================
# Prescribed ocean (single-snapshot FTS, updated each iteration)
# ============================================
# NumericalEarth 0.5.7's `interpolate_state!(…, ::PrescribedOcean, …)` always
# reads time index 1 (no temporal interpolation — upstream TODO). We therefore
# give the component a single-time FTS and write the cyclically-interpolated
# WOA slab into it every iteration (callback below). One-Δt lag, negligible
# against monthly variation.
ocean = PrescribedOcean(grid)  # constant-in-time containers; T=0, S=35, u=v=0

set_prescribed_state!(t) = begin
    copyto!(parent(ocean.sea_surface_temperature[1]), parent(T_woa[Time(mod(t, 365days))]))
    copyto!(parent(ocean.sea_surface_salinity[1]),    parent(S_woa[Time(mod(t, 365days))]))
    return nothing
end
set_prescribed_state!(0.0)

# ============================================
# JRA55 atmosphere, radiation, land (as in omip_forcing, sans sea-ice albedo)
# ============================================
jra55_kw = (; dir = forcing_dir,
              dataset = repeat_year ? RepeatYearJRA55() : MultiYearJRA55(),
              start_date,
              end_date,
              time_indices_in_memory = backend_size)

atmosphere = JRA55PrescribedAtmosphere(arch; jra55_kw...)
radiation  = JRA55PrescribedRadiation(arch; jra55_kw...,
                                      ocean_surface = SurfaceRadiationProperties(0.06, 1.00))
land       = JRA55PrescribedLand(arch; jra55_kw...)

# ============================================
# Coupled model (no sea ice) + simulation
# ============================================
FT = eltype(grid)
atmosphere_ocean_fluxes = flux_config == "corrected" ?
    corrected_atmosphere_ocean_fluxes(FT) :
    ncar_atmosphere_ocean_fluxes(FT)

interfaces = ComponentInterfaces(atmosphere, ocean, nothing;
                                 radiation,
                                 land,
                                 atmosphere_ocean_fluxes,
                                 atmosphere_ocean_velocity_difference = RelativeVelocity())

model = AtmosphereOceanModel(atmosphere, ocean; radiation, land, interfaces)
simulation = Simulation(model; Δt, stop_time)

# WOA update every iteration (runs after time_step! → state used at the NEXT step)
update_prescribed_ocean!(sim) = set_prescribed_state!(time(sim))
add_callback!(simulation, update_prescribed_ocean!, IterationInterval(1))

# ============================================
# Output field handles
# ============================================
ao_fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
net       = model.interfaces.net_fluxes.ocean
rad       = model.radiation.interface_fluxes.ocean
atm_state = model.interfaces.exchanger.atmosphere.state
land_fw   = model.interfaces.exchanger.land.state.freshwater_flux

# name => (field, description, units). Signs/conventions are RAW NumericalEarth
# conventions; postprocess_flux_climatology.jl converts to CMIP conventions.
save_fields = Pair{String, Any}[
    "hfls"    => ao_fluxes.latent_heat,        # W m⁻², + up (ocean cooling)
    "hfss"    => ao_fluxes.sensible_heat,      # W m⁻², + up
    "evs"     => ao_fluxes.water_vapor,        # kg m⁻² s⁻¹, + up
    "rtauuo"  => ao_fluxes.x_momentum,         # ρτˣ, N m⁻² (exchange-grid centers)
    "rtauvo"  => ao_fluxes.y_momentum,         # ρτʸ, N m⁻²
    "ustar"   => ao_fluxes.friction_velocity,  # m s⁻¹
    "JT"      => net.T,                        # net T flux, K m s⁻¹ (turbulent+radiative; Oceananigans BC sign)
    "JS"      => net.S,                        # net S flux, psu m s⁻¹
    "tauuo"   => net.u,                        # kinematic stress, m² s⁻² (Face,Center)
    "tauvo"   => net.v,                        # kinematic stress, m² s⁻² (Center,Face)
    "rlus"    => rad.upwelling_longwave,       # W m⁻², emitted LW, + up   (verified: σT⁴ ≈ 419 @ 293 K)
    "rlds"    => rad.downwelling_longwave,     # W m⁻², absorbed LW, + down (verified in coupled test)
    "rsds"    => rad.downwelling_shortwave,    # W m⁻², transmitted SW, + down = (1−α)·SW↓ (verified)
    "prra"    => atm_state.Jʳⁿ,                # kg m⁻² s⁻¹, + down (rain)
    "prsn"    => atm_state.Jˢⁿ,                # kg m⁻² s⁻¹, + down (snow)
    "friver"  => land_fw,                      # kg m⁻² s⁻¹ (runoff + calving)
    "tos"     => ocean.sea_surface_temperature[1],  # K (prescribed; verification)
    "sos"     => ocean.sea_surface_salinity[1],     # psu
]
# Atmosphere state (debugging / flux decomposition), when present
for (name, sym) in ("Ta" => :T, "qa" => :q, "ua" => :u, "va" => :v, "pa" => :p)
    hasproperty(atm_state, sym) && push!(save_fields, name => getproperty(atm_state, sym))
end

# ============================================
# Monthly-mean accumulator → JLD2
# ============================================
# Robust, writer-free monthly averaging (JLD2Writer expects a standard
# Oceananigans model; PrescribedOcean is not one). Each iteration adds
# field⋅Δt into the current month's bucket; on rollover the mean is written to
#   monthly/<name>/<month_index>   (2-D Float32, no halos)
# with times in monthly/time. Layout is consumed by postprocess_flux_climatology.jl.
const output_path = joinpath(output_dir, run_name * "_monthly_fluxes.jld2")

field_data(f) = interior(f, :, :, 1)
acc_device   = Dict(name => 0 .* similar(field_data(f), Float64) for (name, f) in save_fields)  # device buckets
accumulators = Dict(name => zeros(Float64, size(field_data(f))) for (name, f) in save_fields)   # host staging

cpu_grid = on_architecture(CPU(), grid)
wet_mask = [!inactive_cell(i, j, Nzs, cpu_grid) for i in 1:Nx, j in 1:Ny]

jldopen(output_path, "w") do file
    file["metadata/flux_config"]  = flux_config
    file["metadata/start_year"]   = start_year
    file["metadata/stop_years"]   = stop_years
    file["metadata/dt_seconds"]   = Float64(Δt)
    file["metadata/run_name"]     = run_name
    file["metadata/reference_density"] = Float64(ocean.density)
    file["metadata/heat_capacity"]     = Float64(ocean.heat_capacity)
    file["metadata/descriptions"] = Dict(
        "hfls" => "latent heat flux, W/m2, positive up", "hfss" => "sensible heat flux, W/m2, positive up",
        "evs" => "evaporation mass flux, kg/m2/s, positive up",
        "rtauuo" => "rho*tau_x at cell centers, N/m2", "rtauvo" => "rho*tau_y at cell centers, N/m2",
        "ustar" => "friction velocity, m/s",
        "JT" => "net temperature flux (turb+rad), K m/s, Oceananigans BC sign (positive = cooling)",
        "JS" => "net salinity flux, psu m/s", "tauuo" => "kinematic x-stress, m2/s2", "tauvo" => "kinematic y-stress, m2/s2",
        "rlus" => "emitted longwave, W/m2, positive up", "rlds" => "absorbed longwave, W/m2, positive down",
        "rsds" => "transmitted (net) shortwave (1-alpha)*SWdown, W/m2, positive down",
        "prra" => "rainfall, kg/m2/s, positive down", "prsn" => "snowfall, kg/m2/s, positive down",
        "friver" => "land runoff + calving freshwater flux, kg/m2/s",
        "tos" => "prescribed SST, K", "sos" => "prescribed SSS, psu")
    file["grid/lon"]  = Array(λnodes(cpu_grid, Center(), Center(), Center()))
    file["grid/lat"]  = Array(φnodes(cpu_grid, Center(), Center(), Center()))
    file["grid/wet"]  = wet_mask
end

current_month  = Ref(1)
month_elapsed  = Ref(0.0)

function finalize_month!(m)
    telapsed = month_elapsed[]
    telapsed ≤ 0 && return nothing
    jldopen(output_path, "a+") do file
        for (name, _) in save_fields
            copyto!(accumulators[name], acc_device[name])
            file["monthly/$name/$m"] = Float32.(accumulators[name] ./ telapsed)
        end
        file["monthly/time/$m"] = (m - 0.5) * MONTH
    end
    for (_, a) in acc_device
        fill!(a, 0)
    end
    month_elapsed[] = 0.0
    return nothing
end

function accumulate_fluxes!(sim)
    t = time(sim)
    m = max(1, ceil(Int, t / MONTH))
    if m != current_month[]
        finalize_month!(current_month[])
        current_month[] = m
    end
    dt = sim.Δt
    for (name, f) in save_fields
        acc_device[name] .+= field_data(f) .* dt
    end
    month_elapsed[] += dt
    return nothing
end
add_callback!(simulation, accumulate_fluxes!, IterationInterval(1))

# ============================================
# Progress
# ============================================
wall = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall[])
    # Copy the (small) month-to-date latent-heat bucket to the host before the
    # masked mean: `mean(view(::CuArray, wet_mask))` falls back to scalar getindex,
    # which CUDA disallows.
    hfls_md = Array(acc_device["hfls"])
    Qlat = mean(view(hfls_md, wet_mask)) / max(month_elapsed[], 1.0)
    @info @sprintf("iter %d, t = %s, month %d, ⟨hfls⟩ (month-to-date, wet) = %.1f W/m², wall %.1f s / 100 iter",
                   iteration(sim), prettytime(sim), current_month[], Qlat, elapsed)
    wall[] = time_ns()
end
add_callback!(simulation, progress, IterationInterval(100))

# ============================================
# Run
# ============================================
run!(simulation)
finalize_month!(current_month[])

jldopen(output_path, "a+") do file
    file["metadata/nmonths"] = current_month[]
end
@info "Done. Monthly fluxes written to $output_path ($(current_month[]) months)."
