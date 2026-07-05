# JRA55 × WOA prescribed-ocean air–sea flux comparison

Computes the air–sea fluxes our NumericalEarth/Oceananigans stack produces when the
JRA55(-do) atmosphere blows over a **prescribed** ocean surface given by the WOA23
monthly climatology (no dynamic ocean, no sea ice), on the 1° ORCA surface grid — and
compares them with the flux products published by the CMIP6 **omip2** models
(JRA55-do–forced, like our runs). Design rationale and survey: [PLAN.md](PLAN.md).

## Pipeline

```
1. OMIP2 reference data (anywhere with internet; already done for this repo —
   the processed files are committed in omip_data/omip2_fluxes/)
   julia --project=<repo> fetch_omip2_fluxes.jl all      # pure Julia (+curl)
   # or, downloads only, with no Julia at all:
   ./download_omip2_raw.sh                               # bash+curl, reads omip2_download_urls.txt
   julia --project=<repo> fetch_omip2_fluxes.jl process  # then process in Julia
   → raw ESGF files in $RAW_DIR (default /tmp/omip2_raw, ~4 GB, disposable)
   → processed climatologies in omip_data/omip2_fluxes/<var>_omip2_<MODEL>.nc

2. Our flux run (HPC, 1 GPU; minutes–hours depending on length)
   ./launch_fluxes.sh                     # corrected (COARE 3.6), MultiYearJRA55
   FLUX_CONFIG=ncar ./launch_fluxes.sh    # Large & Yeager (OMIP-2 protocol)
   → <run>_run/<run>_monthly_fluxes.jld2  (raw monthly means, all conventions raw)

   Defaults: START_YEAR=1958, STOP_YEARS=5 — the same MultiYearJRA55 start and
   length as the calibrate_catke_gm_seasonal.jl forward runs, so fluxes are
   directly relatable to the calibration simulations. REPEAT_YEAR=true switches
   to RepeatYearJRA55 (small-download demo mode). The run script loads the
   in-repo OMIPSimulations module standalone (no XESMF/PythonCall needed).

3. Postprocess our run (HPC or anywhere with the run output + repo env)
   julia --project=<repo> postprocess_flux_climatology.jl <run>_monthly_fluxes.jld2
   → <run>_run/climatology/<var>_omip2_NumericalEarth-<config>.nc
   (CMIP names/signs/units; sign conventions verified by runtime assertions;
   CLIM_YEARS=N averages only the last N years — CLIM_YEARS=1 mirrors the
   calibration's last-year-of-run convention)
   Copy/symlink these into omip_data/omip2_fluxes/.

4. Comparison figures + monthly videos
   julia --project=<repo> plot_flux_comparison.jl [extra_data_dir ...]
   → omip_data/omip2_fluxes/figures/fluxcomp_<var>.png (one figure per variable:
     annual-mean maps for every source, ours − OMIP-multi-model-mean, zonal means,
     60°S–60°N seasonal cycle)
   → omip_data/omip2_fluxes/figures/fluxcomp_<var>_monthly.mp4 (same map layout
     animated through the 12 climatological months, with a month cursor on the
     seasonal-cycle panel; skip with VIDEOS=false)
```

## Focus variables

| CMIP name | Meaning | OMIP2 models with data |
|---|---|---|
| `hfls`, `hfss` | latent / sensible heat (+up) | none publish them — derived: `hfls ≈ Lᵥ·evs`, `hfss` as heat-budget residual (CESM2, CMCC) |
| `evs` | evaporation (+up) | CESM2, CMCC-CM2-SR5, EC-Earth3, NorESM2-LM |
| `prra`, `prsn` | rain / snow onto ocean (+down) | CMCC, NorESM2-LM (+EC-Earth3 prsn) |
| `hfds` | net downward heat flux | CESM2, CMCC, CNRM-CM6-1, EC-Earth3, NorESM2-LM, TaiESM1-TIMCOM |
| `rsntds` | net downward shortwave | CESM2, CMCC, CNRM, EC-Earth3, NorESM2-LM |
| `rlntds` | net downward longwave | CESM2, CMCC |
| `wfo` | net water flux into ocean | CMCC, CNRM, EC-Earth3, NorESM2-LM, TaiESM1 |

All OMIP2 climatologies are the **last 360 months of the final JRA55-do cycle**
(matching the existing `omip_data/` tos/t200 maps); our default run window 1988–2018
matches it.

## Design in one paragraph

The coupled model is `AtmosphereOceanModel(JRA55PrescribedAtmosphere, PrescribedOcean;
radiation = JRA55PrescribedRadiation, land = JRA55PrescribedLand)` — every component
prescribed, nothing dynamic. Two small method extensions
([prescribed_ocean_patches.jl](prescribed_ocean_patches.jl)) give `PrescribedOcean`
net-flux fields and route the standard open-water net-flux assembly, so radiation and
`hfds`/`wfo`-type diagnostics work exactly as in production runs. WOA23 monthly SST/SSS
(in-situ, Kelvin) live in a cyclic 12-slab `FieldTimeSeries`; a per-iteration callback
writes the time-interpolated slab into the ocean component (upstream `PrescribedOcean`
does not yet time-interpolate multi-time data). The bulk formulation is the production
`corrected_atmosphere_ocean_fluxes` (COARE 3.6) or `ncar_atmosphere_ocean_fluxes`
(Large & Yeager) from `src/OMIPSimulations/omip_simulation.jl`, with
`RelativeVelocity()` and `BulkTemperature()` — but the ocean is at rest, so stress is
effectively absolute-wind (expect high bias in ACC/WBC regions; see PLAN.md §5).

## Verified conventions (coupled integration test, 2026-07)

A CPU integration test (tiny grid, synthetic atmosphere, 20 °C ocean / 15 °C air /
8 m s⁻¹ wind, 300/350 W m⁻² SW/LW down) confirmed:

- `latent_heat`, `sensible_heat`: **positive up** (193.2 / 62.9 W m⁻²),
- `x_momentum` (ρτˣ): **negative for a westerly** → CMIP `tauuo = −ρτˣ`,
- radiation diagnostics: `upwelling_longwave` +up (σT⁴ = 418.8), `downwelling_longwave`
  and `downwelling_shortwave` stored **positive-down** (350.0 / 282.0 = (1−0.06)·300),
- `hfds = −ρ₀cₚ·J_T` with **exact** closure `hfds = rsntds + rlntds − hfls − hfss`
  (residual 0.0 W m⁻² — no ice, no frazil).

`postprocess_flux_climatology.jl` still asserts these signs at runtime on region means.

## Caveats for interpretation

- **No sea ice**: polar fluxes are open-water fluxes; compare only ice-free regions
  (figures gray out nothing — restrict attention to ~60°S–60°N).
- **Zero surface currents**: relative-wind stress degenerates to absolute wind.
- **Prescribed SST = WOA23 climatology**: no interannual SST variability, so
  flux variability vs year-to-year JRA55 anomalies is atmosphere-driven only; the
  climatological mean is the comparison target.
- **Calendar**: months are 365/12-day climatological months (matching the production
  monthly writers), vs real calendar months in the OMIP output — sub-week phase
  differences in the seasonal cycle are possible.
- **Epoch mismatch**: the default run window (1958–1963, matching the calibration
  forward runs) differs from the OMIP2 climatology epoch (last 360 months ≈
  1988–2018). JRA55-do carries multi-decadal trends, so expect a few W/m² of
  systematic offset from the epoch alone; extend STOP_YEARS (the run is cheap) if
  an epoch-matched comparison is wanted.
- **Model availability** (checked July 2026): the CMCC-CM2-SR5, CNRM-CM6-1, and
  TaiESM1-TIMCOM data nodes are offline/unresolvable with no replicas, so despite
  being in the ESGF index their files are not downloadable. The downloaded
  comparison set (in `omip_data/omip2_fluxes/`) is **CESM2, EC-Earth3,
  NorESM2-LM**: `hfds`/`rsntds`/`evs` ×3 models, `prsn`/`wfo` ×2, `rlntds`/`prra`
  ×1. Latent heat is derived (`Lᵥ·evs`) for all three; sensible heat as the
  heat-budget residual where all four ingredients exist.
