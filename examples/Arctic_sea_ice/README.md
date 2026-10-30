# Pan-Arctic sea ice configuration

Coupled ocean–sea ice simulation of the Arctic on a polar-stereographic grid with GLORYS open boundaries,
adapted from [NoraLoose/PanArcticShenanigans](https://github.com/NoraLoose/PanArcticShenanigans). This is
the forward model for Arctic sea ice calibration.

This example has its own environment (`Project.toml`, `Manifest.toml`), pinned to registered
NumericalEarth 0.8.2 and Oceananigans 0.113, and needs Julia 1.13.

## Requirements

- **Data credentials** as environment variables (`~/API_keys.sh` on the GCP cluster):
  - GLORYS12 (ocean initial and boundary data): `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.
  - ECCO4 (initial sea ice): `ECCO_USERNAME` and `ECCO_WEBDAV_PASSWORD`.
  - JRA55 (atmosphere, rivers) and ETOPO (bathymetry) need no account.

## Running

```bash
cd examples/Arctic_sea_ice
julia +1.13 --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia +1.13 --project=. panarctic.jl [config.toml]
```

On the GCP cluster, from the repository root:

```bash
sbatch examples/Arctic_sea_ice/run_panarctic.slurm [config.toml]
```

Settings (grid, dates, time step, run length, data and output folders) are in `config.toml`. Data folders
default to the caches already on the cluster (`~/GLORYS_data`, `~/JRA55_data`, `~/ECCO_data`). The first run
downloads and inpaints the GLORYS boundary data (cached afterwards). For a quick check, set `stop_hours = 3`.

Output (`arctic_*.jld2`, `*.png`, `*.mp4`) is written to `output_dir`.
