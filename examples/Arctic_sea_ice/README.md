# Pan-Arctic sea ice configuration

> [!WARNING]
> This is an early, uncalibrated snapshot of [PanArcticShenanigans](https://github.com/NoraLoose/PanArcticShenanigans)
> by Nora Loose and Francis Poulin, where the configuration is under active development. Please contribute
> improvements there rather than here.

Coupled ocean–sea ice simulation of the Arctic on a polar-stereographic grid (or, for reference, the
rotated lat-lon grid it replaced upstream; `grid_type` in the script) with GLORYS open boundaries
(oblique radiation for velocities, tracer reservoirs for T and S, Flather for the barotropic transport).
This is the forward model for Arctic sea ice calibration.

This example has its own environment (`Project.toml`): NumericalEarth from its `ss/omip-prototype` branch and
SeawaterPolynomials from its `ss/conversion-functions` branch (the sources the repository root uses), Oceananigans 0.113
and ClimaSeaIce 0.5. It needs Julia 1.13.

## Requirements

- **Data credentials** as environment variables (`~/API_keys.sh` on the GCP cluster):
  - GLORYS12 (initial ocean and sea ice, ocean boundary data): `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.
  - JRA55 (atmosphere, rivers) and GEBCO (bathymetry) need no account.

## Running

```bash
cd examples/Arctic_sea_ice
julia +1.13 --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia +1.13 --project=. panarctic.jl [options]
```

On the GCP cluster, from the repository root:

```bash
sbatch examples/Arctic_sea_ice/run_panarctic.slurm [options]
```

Model settings (grid, resolution, dates, time step, atmosphere) are variables at the top of `panarctic.jl`.
Run options are command-line flags (`--help` lists them): `--stop_hours` (default 120; use 3 for a quick
check), `--output_dir` (default `output`, relative to this directory), `--figures` and `--movies`
(default true), `--architecture` (default GPU), and the data folders `--glorys_dir` and `--jra55_dir`
(default the caches already on the cluster in `$HOME`). The first run downloads and inpaints the GLORYS
data and regrids the bathymetry (both cached afterwards).

Output (`arctic_*.jld2`, `*.png`, `*.mp4`) is written to the output directory.

`analysis/` checks a run against GLORYS: `glorys_ice_series.jl` writes the daily GLORYS sea-ice extent and volume
on the run's grid next to its output and `glorys_surface_series.jl` the daily GLORYS surface T and S (`sbatch
examples/Arctic_sea_ice/analysis/run_glorys_series.slurm <ice or surface> <run dir> <start date> <ndays>`);
`seasonal_analysis.jl <figure dir> <run dir> [<slurm log>]` plots ice extent and volume and the surface means by
region against them, the monthly 3D T and S drift and the daily extrema from the log.

## License

`panarctic.jl` and this README are modified from [PanArcticShenanigans](https://github.com/NoraLoose/PanArcticShenanigans),
Copyright 2026 Nora Loose and Francis Poulin, licensed under the Apache License 2.0 (see `LICENSE` at the repository
root). PanArcticShenanigans is adapted from the
[Barents Sea regional example](https://github.com/NumericalEarth/BjerknesWorkshop/blob/main/tutorials/day2/notebooks/02_barents_sea_regional.ipynb)
by Simone Silvestri. Changes from the source:

- model settings are script variables and run options are command-line flags, in place of `config.toml`;
- grid switch between the polar-stereographic and rotated lat-lon grids; conic domain moved so the closed west edge is land;
- 100 vertical levels with a 1.5 m surface cell; GEBCO 2026 bathymetry, 20 m minimum depth, grid-fitted bottom;
- start on 1993-01-02; daily surface output;
- NumericalEarth from its `ss/omip-prototype` branch;
- oblique radiation for the velocities and tracer reservoirs for T and S, in place of normal radiation, on three
  open edges (Atlantic, Pacific south of the Aleutians, Gulf of Alaska), with the domain extended to y₂ = 3100 km;
- GLORYS boundary data streamed two days at a time; sponge off by default;
- ocean starts from rest; sea ice starts from GLORYS in place of ECCO, with elasto-viscous-plastic dynamics and
  incremental-remapping advection;
- grid floor at 5275 m, the deepest GLORYS level with data; barotropic substeps from the CFL.
