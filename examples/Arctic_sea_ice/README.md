# Pan-Arctic sea ice configuration

Coupled ocean–sea ice simulation of the Arctic on a polar-stereographic grid (or, for reference, the
rotated lat-lon grid it replaced upstream; `grid_type` in the script) with GLORYS open boundaries,
adapted from [NoraLoose/PanArcticShenanigans](https://github.com/NoraLoose/PanArcticShenanigans). This is
the forward model for Arctic sea ice calibration.

This example has its own environment (`Project.toml`), pinned to registered NumericalEarth 0.8.2 and
Oceananigans 0.113, and needs Julia 1.13.

## Requirements

- **Data credentials** as environment variables (`~/API_keys.sh` on the GCP cluster):
  - GLORYS12 (ocean initial and boundary data): `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.
  - ECCO4 (initial sea ice): `ECCO_USERNAME` and `ECCO_WEBDAV_PASSWORD`.
  - JRA55 (atmosphere, rivers) and ETOPO (bathymetry) need no account.

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
(default true), `--architecture` (default GPU), and the data folders `--glorys_dir`, `--jra55_dir`,
`--ecco_dir` (default the caches already on the cluster in `$HOME`). The first run downloads and inpaints
the GLORYS boundary data (cached afterwards).

Output (`arctic_*.jld2`, `*.png`, `*.mp4`) is written to the output directory.
