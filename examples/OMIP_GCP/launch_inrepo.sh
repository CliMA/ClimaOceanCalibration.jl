#!/bin/bash
# GCP a3mega launch script for the in-repo OMIPSimulations submodule.
#
# This is the counterpart of launch.sh that uses the OMIPSimulations module
# now vendored under src/OMIPSimulations/ in this repository, instead of the
# external NumericalEarth/experiments/OMIPSimulations submodule on the
# ss/omip-prototype branch.
#
# Submits an OMIP simulation to SLURM on the GCP a3mega cluster
# (8x H100/node, NVHPC 25.7, HPC-X 2.22.1, juliaup-managed Julia 1.12.3).
#
# Usage:
#   ./launch_inrepo.sh orca                           # ORCA, 1 GPU
#   NCAR=true ./launch_inrepo.sh orca                 # ORCA with NCAR bulk formulae
#   NCAR=true SNOW=true ./launch_inrepo.sh orca       # ORCA + NCAR + snow
#   CB=0.1 NCAR=true ./launch_inrepo.sh orca          # ORCA + NCAR + Cᵇ=0.1
#   KSKEW=1000 KSYMM=500 ./launch_inrepo.sh orca      # custom eddy diffusivities
#   ./launch_inrepo.sh tenthdegree                    # 1/10-degree, 4 GPUs distributed
#   PROFILE=true ./launch_inrepo.sh orca              # nsys-profile run
#   SCRIPT_ONLY=true ./launch_inrepo.sh orca          # write orca.jl, do not submit
#
# OMIPSimulations is a submodule of ClimaOceanCalibration (loaded via
# `using ClimaOceanCalibration.OMIPSimulations`), so the job runs in the
# ClimaOceanCalibration environment with no separate OMIPSimulations dep.
# You can launch from anywhere — no manual `cd` into the depot needed.
#
# Override CLIMA_CALIB_PROJECT=<path> if this script isn't located inside
# ClimaOceanCalibration.jl.
#
# Credentials (e.g. ECCO_USERNAME, ECCO_WEBDAV_PASSWORD) are loaded from
# ~/API_keys.sh inside the job if that file exists.
#
# Setup (one-time, on a fresh checkout):
#   1. Clone NumericalEarth somewhere local, e.g.
#        git clone https://github.com/CliMA/NumericalEarth.jl ~/.julia/dev/NumericalEarth
#   2. From the ClimaOceanCalibration.jl project root, dev NumericalEarth
#      into this project so Pkg can resolve it:
#        julia +1.12.3 --project=. -e '
#          using Pkg
#          Pkg.develop(path="<path to NumericalEarth>")
#          Pkg.instantiate()'
#   3. Ensure juliaup has channel 1.12.3 installed:
#        juliaup add 1.12.3
#   4. Ensure $HOME/env_nvhpc_25.7.sh exists on the GCP a3mega cluster
#      (provides NVHPC 25.7 / CUDA 12.9 / HPC-X 2.22.1).

set -euo pipefail

STOP_YEARS="${STOP_YEARS:-300}"

# ── Locate the ClimaOceanCalibration project ──────────────────────────
# launch_inrepo.sh lives at examples/OMIP_GCP/launch_inrepo.sh — the
# project root is two levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIMA_CALIB_PROJECT="${CLIMA_CALIB_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ ! -f "$CLIMA_CALIB_PROJECT/src/OMIPSimulations/OMIPSimulations.jl" ]]; then
    echo "Error: in-repo OMIPSimulations not found at '$CLIMA_CALIB_PROJECT/src/OMIPSimulations/'." >&2
    echo "       Set CLIMA_CALIB_PROJECT to the ClimaOceanCalibration.jl root." >&2
    exit 1
fi

usage() {
    cat <<'USAGE'
Usage: ./launch_inrepo.sh <config> [extra sbatch args...]

GCP a3mega variant that uses the in-repo OMIPSimulations submodule
(src/OMIPSimulations/). Runs in the ClimaOceanCalibration project
environment. Override CLIMA_CALIB_PROJECT=<path> if launch_inrepo.sh
is not sitting inside ClimaOceanCalibration.jl.

Configurations:
  halfdegree      Half-degree TripolarGrid           (1 GPU)
  orca            ORCA grid                          (1 GPU)
  tenthdegree     1/10-degree TripolarGrid           (4 GPUs, distributed)

Environment variables (physics):
  NCAR          Set to "true" for OMIP-2/NCAR bulk formulae
  CORRECTED     Set to "true" for corrected COARE 3.6 fluxes
  SNOW          Set to "true" to enable snow thermodynamics
  ICE_DYNAMICS  Set to "false" to disable sea-ice dynamics (thermo-only ice).
                Default: true.
  SKIN_TEMPERATURE  Set to "true" to compute the atmosphere-ocean interface
                temperature as a flux-balance "skin" temperature
                (SkinTemperature(DiffusiveFlux(δ, 1e-2)), with δ = half the
                top grid cell's thickness) instead of using the bulk
                temperature of the top ocean cell (BulkTemperature, the
                default). Default: false.
  KSKEW         Isopycnal skew diffusivity κ_skew (default: per-config; 0 = off)
  KSYMM         Isopycnal symmetric diffusivity κ_symmetric (default: per-config; 0 = off)
  BIHARMONIC    Biharmonic viscosity timescale (default: per-config; "nothing" = off)
  BIHVISC       Constant biharmonic viscosity ν in m^4/s (default: unset).
  CB            CATKE buoyancy mixing length parameter Cᵇ (default: 0.28).
                Forwarded as catke_parameters = (; mixing_length = (; Cᵇ = …)).
  CLOSURE       Ocean vertical closure: "catke" (default), "simple", "nori",
                "rbvd", "kpp", "nemo_tke"
  WIND_VELOCITY Set to "true" to use absolute wind in the bulk formula
                (Δu = u_atm) instead of OMIP-2 relative wind (Δu = u_atm − u_ocean)
  VONKARMAN_SCALING  Scaling factor on the von Kármán constant (0.4) in the
                atmosphere–ocean bulk fluxes; effective κ = 0.4 * VONKARMAN_SCALING.
                Default: 1 (unmodified). Only affects CORRECTED / SHEAR_GUST runs.
  DZ_TOP        Target thickness of the top (surface) cell in meters

Equatorial-MLD knobs (closure parameters; configuration switches):
  SHEAR_GUST    Mahrt–Sun/Edson shear-aware gustiness; implies :corrected
  MIN_SALINITY  Floor (psu) below which freshening freshwater flux is suppressed.
                Default: 1.
  NORMALIZE_SALINITY Set to "true" to normalize salinity flux. Default: false.
  CATKE_CWUSTAR `Cᵂu★` of CATKEEquation. Default (Oceananigans): 3.179.
                Forwarded as catke_parameters = (; tke_equation = (; Cᵂu★ = …)).

Advanced parameter overrides:
  CATKE_PARAMS_EXPR  Verbatim Julia NamedTuple appended to catke_parameters.
                     Example: CATKE_PARAMS_EXPR='(; mixing_length = (; Cˢ = 0.7))'
  GM_PARAMS_EXPR     Verbatim Julia NamedTuple appended to gm_parameters.
                     Example: GM_PARAMS_EXPR='(; slope_limiter = (; max_slope = 0.01))'

Environment variables (I/O & runtime):
  BACKEND_SIZE  Number of JRA55 time indices kept in memory (default: 240)
  FORCING_DIR   Path to JRA55 forcing data
                (default: /home/ext_xinkai_caltech_edu/JRA55_data)
  RESTORING_DIR Path to WOA/ECCO restoring and initialization data
                (default: /home/ext_xinkai_caltech_edu/ECCO_data)
  STAGING_DIR   Base directory for JRA55 staging (default: ./staged_data)
  NO_STAGING    Set to "true" to disable JRA55 staging entirely (read directly
                from FORCING_DIR, no staging callback). Default: false.

Diagnostic knobs (for isolating the ~1600-day forcing blowup):
  BACKEND_SIZE  See above. Sweep (e.g. 60, 120) to test whether the blowup time
                shifts with the in-memory window → backend/index bug, or stays
                fixed → data content / absolute-time arithmetic.
  REPEAT_YEAR   Set to "true" to use RepeatYearJRA55 (single repeating year)
                instead of MultiYearJRA55. If the blowup persists, it is pure
                time-index arithmetic; if it vanishes, it is the multi-year data
                content or a year-file/staging transition. Default: false.
  NO_STAGING    See above. Rules the staging callback in/out as the corruptor.
  PREFETCH      Set to "false" to load JRA55 windows synchronously instead of via
                the background PrefetchingBackend. DECISIVE test: if the blowup
                vanishes with PREFETCH=false, it is the prefetch path. To then
                separate race-vs-logic, rerun PREFETCH=true with THREADS=1
                (single thread makes the prefetch @spawn synchronous). Default: true.
  STOP_YEARS    Simulation length in years (default: 300)
  THREADS       Number of Julia threads / CPUs per task (default: 4)
  LOG_DIR       Directory for SLURM logs (default: <project>/logs)
  INSTANTIATE   Run Pkg.instantiate() in the job before launch (default: true)
  PROFILE       Set to "true" for nsys profiling (also disables diagnostics
                and drops `pickup` from `run!`)
  SCRIPT_ONLY   Set to "true" to write ./<RUN_NAME>.jl and exit without
                submitting to SLURM. Default: false. Run the script yourself
                with juliaup.
  VISUALIZE     Set to "true" to render OMIP diagnostic figures after run!
                completes, in the same SLURM job. Figures are saved to
                <RUN_NAME>_run/figures/. Default: false.
  YEARS_FROM_END  Averaging window (years from the last snapshot) used by the
                  post-run visualization. Default: 2.
  FIG           Figure selection forwarded to visualize_omip.jl when
                VISUALIZE=true (e.g. "1,4,17" or "14-19"). Default: all.
  THEME         Figure theme for the post-run visualization ("light"/"dark").
                Default: light.

GCP-specific:
  PARTITION     Default: a3mega
  TIME          Default: 10:00:00
  MEM           Default: 200GB

Examples:
  ./launch_inrepo.sh orca
  NCAR=true ./launch_inrepo.sh orca
  CB=0.1 NCAR=true ./launch_inrepo.sh orca
  KSKEW=0 ./launch_inrepo.sh orca                    # disable eddy closure
  BIHARMONIC=nothing ./launch_inrepo.sh orca         # disable biharmonic viscosity
  DZ_TOP=2 ./launch_inrepo.sh orca
  SHEAR_GUST=true ./launch_inrepo.sh orca
  SKIN_TEMPERATURE=true ./launch_inrepo.sh orca      # flux-balance skin temperature
  VONKARMAN_SCALING=0.9 CORRECTED=true ./launch_inrepo.sh orca   # κ = 0.36 sensitivity run
  CATKE_CWUSTAR=5.0 ./launch_inrepo.sh orca
  CATKE_PARAMS_EXPR='(; mixing_length = (; Cˢ = 0.7))' ./launch_inrepo.sh orca
  GM_PARAMS_EXPR='(; slope_limiter = (; max_slope = 0.01))' ./launch_inrepo.sh orca
  FORCING_DIR=/other/jra55 RESTORING_DIR=/other/ecco STAGING_DIR=/scratch/staged ./launch_inrepo.sh orca
  PROFILE=true ./launch_inrepo.sh orca
  SCRIPT_ONLY=true ./launch_inrepo.sh orca           # emit orca.jl, no sbatch
  ./launch_inrepo.sh tenthdegree
USAGE
}

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    usage
    exit 1
fi
shift || true

case "$CONFIG" in
    halfdegree|half_degree)
        CONFIG="halfdegree"
        ;;
    orca|tenthdegree) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown configuration '$CONFIG'" >&2
        usage
        exit 1
        ;;
esac

# ── Per-config defaults ───────────────────────────────────────────────
case "$CONFIG" in
    halfdegree)
        DEFAULT_KSKEW=250;  DEFAULT_KSYMM=100; NZ=70;  DEFAULT_DT="25minutes"; DEFAULT_DZ_TOP=""
        DEFAULT_BIHARMONIC="40days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = ${STOP_YEARS} * 365days
run!(sim, pickup=:latest)"
        ;;
    orca)
        DEFAULT_KSKEW=500;  DEFAULT_KSYMM=250; NZ=70;  DEFAULT_DT="30minutes"; DEFAULT_DZ_TOP=""
        DEFAULT_BIHARMONIC="10days"; ARCH="GPU()"; GPUS_PER_NODE=1
        EXTRA_USING=""; FILE_SPLIT=""
        RUN_CMD="sim.stop_time = ${STOP_YEARS} * 365days
run!(sim; pickup = :latest)"
        ;;
    tenthdegree)
        DEFAULT_KSKEW=0;    DEFAULT_KSYMM=0;   NZ=100; DEFAULT_DT="2minutes";  DEFAULT_DZ_TOP="2.5"
        DEFAULT_BIHARMONIC="nothing"; ARCH="Distributed(GPU(), partition=Partition(1, 4))"; GPUS_PER_NODE=4
        EXTRA_USING="using Oceananigans.DistributedComputations"
        FILE_SPLIT="file_splitting_interval = 180days,"
        RUN_CMD="sim.stop_time = 91days
run!(sim)

sim.Δt = 10minutes
sim.stop_time = ${STOP_YEARS} * 365days
run!(sim; pickup = true)"
        ;;
esac

# Profile mode: drop `pickup` from `run!` so the simulation starts from
# scratch (resuming a checkpoint skips the init phase the profiler needs).
if [[ "${PROFILE:-false}" == "true" ]]; then
  RUN_CMD="sim.stop_iteration = 200; run!(sim)"
fi

# FRESH mode: strip `pickup` so the run starts from 1958 (iteration 0). Needed to
# reproduce the ~1568-day blowup, which depends on the continuously-evolved FTS
# window alignment — `pickup=:latest` re-initializes the windows and DODGES the
# bug. Tagged `_fresh` so it gets its own output dir (no checkpoint collision).
if [[ "${FRESH:-false}" == "true" ]]; then
  RUN_CMD="${RUN_CMD//, pickup = :latest/}"
  RUN_CMD="${RUN_CMD//; pickup = :latest/}"
  RUN_CMD="${RUN_CMD//; pickup = true/}"
  RUN_CMD="${RUN_CMD//pickup = :latest/}"
fi

# 0 means "no eddy closure" (maps to Julia `nothing`)
export KSKEW="${KSKEW:-$DEFAULT_KSKEW}"
export KSYMM="${KSYMM:-$DEFAULT_KSYMM}"
export DT="${DT:-$DEFAULT_DT}"
export DZ_TOP="${DZ_TOP:-$DEFAULT_DZ_TOP}"
export BIHARMONIC="${BIHARMONIC:-$DEFAULT_BIHARMONIC}"
KSKEW_JULIA="$KSKEW"; [[ "$KSKEW" == "0" ]] && KSKEW_JULIA="nothing"
KSYMM_JULIA="$KSYMM"; [[ "$KSYMM" == "0" ]] && KSYMM_JULIA="nothing"
export KSKEW_JULIA KSYMM_JULIA
export NZ DT ARCH EXTRA_USING FILE_SPLIT RUN_CMD

# ── Build run name from config + options ──────────────────────────────
RUN_NAME="$CONFIG"
[[ "${CORRECTED:-false}" == "true" ]]          && RUN_NAME="${RUN_NAME}_corrected"
[[ "${NCAR:-false}" == "true" ]]               && RUN_NAME="${RUN_NAME}_ncar"
[[ "${SNOW:-false}" == "true" ]]               && RUN_NAME="${RUN_NAME}_snow"
[[ "${SKIN_TEMPERATURE:-false}" == "true" ]]   && RUN_NAME="${RUN_NAME}_skintemp"
[[ "${ICE_DYNAMICS:-true}" == "false" ]]       && RUN_NAME="${RUN_NAME}_noicedyn"
[[ "${CLOSURE:-catke}" == "simple"   ]]        && RUN_NAME="${RUN_NAME}_simple"
[[ "${CLOSURE:-catke}" == "nori"     ]]        && RUN_NAME="${RUN_NAME}_nori"
[[ "${CLOSURE:-catke}" == "rbvd"     ]]        && RUN_NAME="${RUN_NAME}_rbvd"
[[ "${CLOSURE:-catke}" == "kpp"      ]]        && RUN_NAME="${RUN_NAME}_kpp"
[[ "${CLOSURE:-catke}" == "nemo_tke" ]]        && RUN_NAME="${RUN_NAME}_nemotke"
[[ "${WIND_VELOCITY:-false}" == "true" ]]      && RUN_NAME="${RUN_NAME}_wind"
[[ "${VONKARMAN_SCALING:-1}" != "1" ]]         && RUN_NAME="${RUN_NAME}_vk${VONKARMAN_SCALING}"
[[ "${NORMALIZE_SALINITY:-false}" == "true" ]] && RUN_NAME="${RUN_NAME}_normsalt"
[[ -n "${CB:-}" ]]                             && RUN_NAME="${RUN_NAME}_cb${CB}"
[[ "$KSKEW" != "$DEFAULT_KSKEW" ]]             && RUN_NAME="${RUN_NAME}_kskew${KSKEW}"
[[ "$KSYMM" != "$DEFAULT_KSYMM" ]]             && RUN_NAME="${RUN_NAME}_ksymm${KSYMM}"
[[ "$BIHARMONIC" != "$DEFAULT_BIHARMONIC" ]]   && RUN_NAME="${RUN_NAME}_bih${BIHARMONIC}"
[[ -n "${BIHVISC:-}" ]]                        && RUN_NAME="${RUN_NAME}_bihvisc${BIHVISC}"
[[ "$DZ_TOP" != "$DEFAULT_DZ_TOP" ]]           && RUN_NAME="${RUN_NAME}_dz${DZ_TOP}"
[[ "${SHEAR_GUST:-false}" == "true" ]]         && RUN_NAME="${RUN_NAME}_sgust"
[[ -n "${CATKE_CWUSTAR:-}" ]]                  && RUN_NAME="${RUN_NAME}_cwu${CATKE_CWUSTAR}"
[[ -n "${MIN_SALINITY:-}" ]]                   && RUN_NAME="${RUN_NAME}_smin${MIN_SALINITY}"
[[ -n "${CATKE_PARAMS_EXPR:-}" ]]              && RUN_NAME="${RUN_NAME}_catkeext"
[[ -n "${GM_PARAMS_EXPR:-}" ]]                 && RUN_NAME="${RUN_NAME}_gmext"
# Diagnostic knobs — tag so each run gets its own output dir / checkpoint and
# pickup=:latest never resumes a different run's state.
[[ -n "${BACKEND_SIZE:-}" ]]                   && RUN_NAME="${RUN_NAME}_be${BACKEND_SIZE}"
[[ "${REPEAT_YEAR:-false}" == "true" ]]        && RUN_NAME="${RUN_NAME}_repeatyr"
[[ "${NO_STAGING:-false}" == "true" ]]         && RUN_NAME="${RUN_NAME}_nostage"
[[ "${PREFETCH:-true}" == "false" ]]           && RUN_NAME="${RUN_NAME}_noprefetch"
[[ "${FRESH:-false}" == "true" ]]              && RUN_NAME="${RUN_NAME}_fresh"
[[ "$STOP_YEARS" != "300" ]]                   && RUN_NAME="${RUN_NAME}_${STOP_YEARS}yr"

REPORT_NAME="${REPORT_NAME:-${RUN_NAME}_report}"
JOB_NAME="${JOB_NAME:-$RUN_NAME}"
LOG_DIR="${LOG_DIR:-${CLIMA_CALIB_PROJECT}/logs}"
export REPORT_NAME

# ── Build the Julia script (used by both SCRIPT_ONLY and SLURM paths) ─
# All Julia-side variable defaults live here, in the outer shell, so the
# same JULIA_EXPR is reproducible without sbatch.
FORCING_DIR="${FORCING_DIR:-/home/ext_xinkai_caltech_edu/JRA55_data}"
RESTORING_DIR="${RESTORING_DIR:-/home/ext_xinkai_caltech_edu/ECCO_data}"
STAGING_DIR="${STAGING_DIR:-./staged_data}"
NO_STAGING="${NO_STAGING:-false}"
REPEAT_YEAR="${REPEAT_YEAR:-false}"
PREFETCH="${PREFETCH:-true}"
PROBE="${PROBE:-false}"
# NO_STAGING=true wins over STAGING_DIR: blank it so the no-staging branch below
# reads JRA55 directly from FORCING_DIR (rules the staging callback in/out).
# REPEAT_YEAR also forces no-staging: the staging callback stages MultiYear
# yearly files and is meaningless / breaks for the single RepeatYear file.
{ [[ "$NO_STAGING" == "true" ]] || [[ "$REPEAT_YEAR" == "true" ]]; } && STAGING_DIR=""
CB="${CB:-}"
BIHVISC="${BIHVISC:-}"
DZ_TOP="${DZ_TOP:-}"
SHEAR_GUST="${SHEAR_GUST:-false}"
CATKE_CWUSTAR="${CATKE_CWUSTAR:-}"
MIN_SALINITY="${MIN_SALINITY:-}"
BACKEND_SIZE="${BACKEND_SIZE:-}"
NCAR="${NCAR:-false}"
CORRECTED="${CORRECTED:-false}"
SNOW="${SNOW:-false}"
ICE_DYNAMICS="${ICE_DYNAMICS:-true}"
SKIN_TEMPERATURE="${SKIN_TEMPERATURE:-false}"
NORMALIZE_SALINITY="${NORMALIZE_SALINITY:-false}"
VONKARMAN_SCALING="${VONKARMAN_SCALING:-1}"
CATKE_PARAMS_EXPR="${CATKE_PARAMS_EXPR:-}"
GM_PARAMS_EXPR="${GM_PARAMS_EXPR:-}"

# Build catke_parameters / gm_parameters NamedTuples.
CATKE_ML_NT="(;)"
[[ -n "$CB" ]] && CATKE_ML_NT="(; Cᵇ = ${CB})"

CATKE_TKE_NT="(;)"
[[ -n "$CATKE_CWUSTAR" ]] && CATKE_TKE_NT="(; Cᵂu★ = ${CATKE_CWUSTAR})"

CATKE_PARAMS_NT="merge((; mixing_length = ${CATKE_ML_NT}, tke_equation = ${CATKE_TKE_NT}), ${CATKE_PARAMS_EXPR:-(;)})"

# κ_skew = 0 / κ_symmetric = 0 → eddy closure disabled (handled inside
# omip_closure). nothing is also accepted as a sentinel.
GM_BASE_NT="(; κ_skew = ${KSKEW_JULIA}, κ_symmetric = ${KSYMM_JULIA})"
GM_PARAMS_NT="merge(${GM_BASE_NT}, ${GM_PARAMS_EXPR:-(;)})"

# ── Build optional kwargs strings ─────────────────────────────────────
STAGING_KWARG=""
if [[ -n "$STAGING_DIR" ]]; then
    RUN_STAGING_DIR="${STAGING_DIR}/${RUN_NAME}"
    STAGING_KWARG="staging_dir = \"${RUN_STAGING_DIR}\","
fi

BIHVISC_KWARG=""
[[ -n "$BIHVISC" ]] && BIHVISC_KWARG="biharmonic_viscosity = ${BIHVISC},"

DZ_TOP_KWARG=""
[[ -n "$DZ_TOP" ]] && DZ_TOP_KWARG="Δz_top = ${DZ_TOP},"

MIN_SALINITY_KWARG=""
[[ -n "$MIN_SALINITY" ]] && MIN_SALINITY_KWARG="ocean_minimum_salinity = ${MIN_SALINITY},"

NORMALIZE_SALINITY_KWARG=""
[[ "$NORMALIZE_SALINITY" == "true" ]] && NORMALIZE_SALINITY_KWARG="normalize_salinity = true,"

BACKEND_KWARG=""
[[ -n "$BACKEND_SIZE" ]] && BACKEND_KWARG="backend_size = ${BACKEND_SIZE},"

REPEAT_YEAR_KWARG=""
[[ "$REPEAT_YEAR" == "true" ]] && REPEAT_YEAR_KWARG="repeat_year_forcing = true,"

PREFETCH_KWARG=""
[[ "$PREFETCH" == "false" ]] && PREFETCH_KWARG="prefetch = false,"

# PROBE: inject the blowup probe (per-field window/index/NaN report at the
# failing step) right after sim construction. Does NOT change RUN_NAME, so
# pickup=:latest resumes the existing run's checkpoint and reaches iter ~75264
# in minutes instead of hours.
PROBE_LINE=""
if [[ "$PROBE" == "true" ]]; then
    # Probe auto-detects the NaN onset, so the window is only a coarse pre-print
    # range. Defaults bracket both the fresh (~75266) and pickup (~76130) blowups.
    PROBE_FIRST="${PROBE_FIRST:-75200}"
    PROBE_LAST="${PROBE_LAST:-76200}"
    PROBE_LINE="include(\"${CLIMA_CALIB_PROJECT}/examples/OMIP_GCP/blowup_probe.jl\")
add_blowup_probe!(sim; first_iter = ${PROBE_FIRST}, last_iter = ${PROBE_LAST})

"
fi

FLUX_KWARG=""
[[ "$NCAR" == "true" ]]        && FLUX_KWARG="flux_configuration = :ncar,"
[[ "$CORRECTED" == "true" ]]   && FLUX_KWARG="flux_configuration = :corrected,"
[[ "$SHEAR_GUST" == "true" ]]  && FLUX_KWARG="flux_configuration = :shear_aware,"

CLOSURE_KWARG=""
[[ "${CLOSURE:-catke}" == "simple"   ]] && CLOSURE_KWARG="vertical_closure = :simple,"
[[ "${CLOSURE:-catke}" == "nori"     ]] && CLOSURE_KWARG="vertical_closure = :nori,"
[[ "${CLOSURE:-catke}" == "rbvd"     ]] && CLOSURE_KWARG="vertical_closure = :rbvd,"
[[ "${CLOSURE:-catke}" == "kpp"      ]] && CLOSURE_KWARG="vertical_closure = :kpp,"
[[ "${CLOSURE:-catke}" == "nemo_tke" ]] && CLOSURE_KWARG="vertical_closure = :nemo_tke,"

VELOCITY_KWARG=""
[[ "${WIND_VELOCITY:-false}" == "true" ]] && VELOCITY_KWARG="velocity_formulation = :wind,"

# von Kármán scaling — only meaningful for the COARE path (:corrected / :shear_aware).
VONKARMAN_KWARG=""
[[ "$VONKARMAN_SCALING" != "1" ]] && VONKARMAN_KWARG="von_karman_scaling = ${VONKARMAN_SCALING},"

SNOW_KWARG=""
[[ "$SNOW" == "true" ]] && SNOW_KWARG="with_snow = true,"

SKIN_TEMPERATURE_KWARG=""
[[ "$SKIN_TEMPERATURE" == "true" ]] && SKIN_TEMPERATURE_KWARG="skin_temperature = true,"

ICE_DYNAMICS_KWARG=""
[[ "$ICE_DYNAMICS" == "false" ]] && ICE_DYNAMICS_KWARG="with_ice_dynamics = false,"

DIAGNOSTICS_KWARG=""
[[ "${PROFILE:-false}" == "true" ]] && DIAGNOSTICS_KWARG="diagnostics = false,"

# ── Build Julia expression ────────────────────────────────────────────
# Use the in-repo OMIPSimulations submodule rather than an external package.
JULIA_EXPR="using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations
using Oceananigans
using Oceananigans.Units
using CUDA
${EXTRA_USING}

sim = omip_simulation(:${CONFIG};
                      arch = ${ARCH},
                      Nz = ${NZ},
                      depth = 5500,
                      ${DZ_TOP_KWARG}
                      catke_parameters = ${CATKE_PARAMS_NT},
                      gm_parameters    = ${GM_PARAMS_NT},
                      biharmonic_timescale = ${BIHARMONIC},
                      ${BIHVISC_KWARG}
                      ${FLUX_KWARG}
                      ${CLOSURE_KWARG}
                      ${VELOCITY_KWARG}
                      ${VONKARMAN_KWARG}
                      ${SNOW_KWARG}
                      ${SKIN_TEMPERATURE_KWARG}
                      ${ICE_DYNAMICS_KWARG}
                      ${DIAGNOSTICS_KWARG}
                      ${MIN_SALINITY_KWARG}
                      ${NORMALIZE_SALINITY_KWARG}
                      Δt = ${DT},
                      forcing_dir = \"${FORCING_DIR}\",
                      restoring_dir = \"${RESTORING_DIR}\",
                      ${STAGING_KWARG}
                      ${BACKEND_KWARG}
                      ${REPEAT_YEAR_KWARG}
                      ${PREFETCH_KWARG}
                      ${FILE_SPLIT}
                      output_dir = \"${RUN_NAME}_run\",
                      filename_prefix = \"${RUN_NAME}\")

${PROBE_LINE}${RUN_CMD}"

THREADS="${THREADS:-4}"
export THREADS

# Post-run visualization knobs (consumed inside the SLURM job).
export VISUALIZE="${VISUALIZE:-false}"
export YEARS_FROM_END="${YEARS_FROM_END:-2}"
export FIG="${FIG:-all}"
export THEME="${THEME:-light}"

# ── SCRIPT_ONLY: emit ./<RUN_NAME>.jl and exit ────────────────────────
if [[ "${SCRIPT_ONLY:-false}" == "true" ]]; then
    SCRIPT_PATH="./${RUN_NAME}.jl"
    GEN_DATE="$(date)"
    MPIRUN_HINT_LINE="#"
    if (( GPUS_PER_NODE > 1 )); then
        MPIRUN_HINT_LINE="# For this multi-rank config (${GPUS_PER_NODE} ranks), prepend: mpirun -np ${GPUS_PER_NODE}"
    fi
    cat > "$SCRIPT_PATH" <<EOF_SCRIPT
# Auto-generated by launch_inrepo.sh on ${GEN_DATE}
# Config: ${CONFIG}, Run: ${RUN_NAME}
# Run with:
#   julia +1.12.3 --project=${CLIMA_CALIB_PROJECT} --check-bounds=no -t ${THREADS} ${SCRIPT_PATH}
${MPIRUN_HINT_LINE}

${JULIA_EXPR}
EOF_SCRIPT
    echo "Wrote $SCRIPT_PATH"
    echo "Run with:"
    echo "  julia +1.12.3 --project=$CLIMA_CALIB_PROJECT --check-bounds=no -t ${THREADS} $SCRIPT_PATH"
    if (( GPUS_PER_NODE > 1 )); then
        echo "  (or: mpirun -np ${GPUS_PER_NODE} julia +1.12.3 --project=$CLIMA_CALIB_PROJECT --check-bounds=no -t ${THREADS} $SCRIPT_PATH)"
    fi
    exit 0
fi

# ── SLURM path ────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# Write the Julia script to LOG_DIR; the SLURM job runs it directly with
# `julia ... $SCRIPT_PATH` instead of `julia -e "$JULIA_EXPR"`. Same code,
# but easier to debug (you can cat the script that ran).
SCRIPT_PATH="${LOG_DIR}/${RUN_NAME}.jl"
GEN_DATE="$(date)"
cat > "$SCRIPT_PATH" <<EOF_SCRIPT
# Auto-generated by launch_inrepo.sh on ${GEN_DATE}
# Config: ${CONFIG}, Run: ${RUN_NAME}

${JULIA_EXPR}
EOF_SCRIPT
export SCRIPT_PATH

SBATCH_ARGS=()
PARTITION="${PARTITION:-a3mega}"

SBATCH_ARGS+=(--gres="gpu:${GPUS_PER_NODE}")
SBATCH_ARGS+=(--ntasks-per-node="${GPUS_PER_NODE}")

SBATCH_ARGS+=(--cpus-per-task="${THREADS}")

SBATCH_ARGS+=(--partition="${PARTITION}")

TIME="${TIME:-10:00:00}"
SBATCH_ARGS+=(--time="${TIME}")

MEM="${MEM:-200GB}"
SBATCH_ARGS+=(--mem="${MEM}")

if [[ "${PROFILE:-false}" == "true" ]]; then
    SBATCH_ARGS+=(-o "${LOG_DIR}/${RUN_NAME}_profile.%j.out")
    SBATCH_ARGS+=(-e "${LOG_DIR}/${RUN_NAME}_profile.%j.out")
    SBATCH_ARGS+=(-J "${JOB_NAME}_profile")
    SBATCH_ARGS+=(--export="ALL,PROFILE=true,REPORT_NAME=${REPORT_NAME},CONFIG=${CONFIG},RUN_NAME=${RUN_NAME},CLIMA_CALIB_PROJECT=${CLIMA_CALIB_PROJECT},SCRIPT_PATH=${SCRIPT_PATH}")
else
    SBATCH_ARGS+=(-o "${LOG_DIR}/${RUN_NAME}.%j.out")
    SBATCH_ARGS+=(-e "${LOG_DIR}/${RUN_NAME}.%j.out")
    SBATCH_ARGS+=(-J "$JOB_NAME")
    SBATCH_ARGS+=(--export="ALL,CONFIG=${CONFIG},RUN_NAME=${RUN_NAME},CLIMA_CALIB_PROJECT=${CLIMA_CALIB_PROJECT},SCRIPT_PATH=${SCRIPT_PATH}")
fi

sbatch "${SBATCH_ARGS[@]}" "$@" <<'EOF'
#!/bin/bash
#SBATCH -N 1

# ── GCP a3mega environment ────────────────────────────────────────────
# NVHPC 25.7, CUDA 12.9, HPC-X 2.22.1 — set up by env_nvhpc_25.7.sh.
source "$HOME/env_nvhpc_25.7.sh"

# Julia / CUDA settings
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NVTX_CALLBACKS=gc
export DATADEPS_ALWAYS_ACCEPT=true

# Optional API keys (ECCO credentials etc.)
[ -f "$HOME/API_keys.sh" ] && source "$HOME/API_keys.sh"

THREADS="${THREADS:-4}"
INSTANTIATE="${INSTANTIATE:-true}"

if [[ "$INSTANTIATE" == "true" ]]; then
    julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --startup-file=no -e 'using Pkg; Pkg.instantiate()'
fi

# Pin Julia 1.12.3 via juliaup channel selector. No mpirun for single-GPU
# runs (halfdegree, orca); tenthdegree uses mpirun to launch the 4 MPI
# ranks needed by Distributed(GPU(), partition=Partition(1, 4)).
NRANKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
if [[ "${PROFILE:-false}" == "true" ]]; then
    echo "Profiling ${RUN_NAME} -> ${REPORT_NAME}"
    if (( NRANKS > 1 )); then
        mpirun -np "$NRANKS" \
            nsys profile --trace=cuda \
                --output="$REPORT_NAME" --force-overwrite true \
                julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "${THREADS}" "$SCRIPT_PATH"
    else
        nsys profile --trace=cuda \
            --output="$REPORT_NAME" --force-overwrite true \
            julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "${THREADS}" "$SCRIPT_PATH"
    fi
else
    if (( NRANKS > 1 )); then
        mpirun -np "$NRANKS" \
            julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "${THREADS}" "$SCRIPT_PATH"
    else
        julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --check-bounds=no -t "${THREADS}" "$SCRIPT_PATH"
    fi
fi

# ── Optional post-run visualization (VISUALIZE=true) ──────────────────
# Render OMIP diagnostic figures for the run that just finished. Runs once
# (outside any mpirun) and saves into <RUN_NAME>_run/figures/. The simulation
# wrote its output to "${RUN_NAME}_run" relative to this job's CWD, so the
# visualize prefix is that same path without the `_run` suffix. visualize_omip.jl
# auto-derives the output dir as "<RUN_PREFIX>_run/figures", so no output-dir arg
# is needed here.
if [[ "${VISUALIZE:-false}" == "true" ]]; then
    echo "Rendering OMIP diagnostic figures for ${RUN_NAME} -> ${RUN_NAME}_run/figures"
    RUN_PREFIX="$(pwd)/${RUN_NAME}" \
    RUN_LABEL="${RUN_NAME}" \
    YEARS_FROM_END="${YEARS_FROM_END:-2}" \
    FIG="${FIG:-all}" \
    THEME="${THEME:-light}" \
        julia +1.12.3 --project="$CLIMA_CALIB_PROJECT" --startup-file=no -t "${THREADS}" \
            "$CLIMA_CALIB_PROJECT/examples/OMIP_GCP/visualize_omip.jl"
fi
EOF
