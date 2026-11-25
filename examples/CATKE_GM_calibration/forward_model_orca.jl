# forward_model_orca.jl
# Per-member forward model: runs ClimaOcean's `OneDegree()` OMIP configuration on
# the eORCA1 mesh with its CATKE and GM coefficients multiplied by the calibration
# scalings, with the built-in diagnostics disabled and a small set of custom
# output writers attached:
#
#   - <N>year_average.jld2   : single end-of-run mean of T, S, u on the full
#                              ORCA grid, over the final `sampling_window`
#                              (N = window in years, e.g. 5year_average.jld2).
#   - global_means.jld2      : 30-day averaged global-mean T and S (scalars).
#   - horizontal_means.jld2  : 30-day averaged horizontal-mean T and S (1D in z).
#   - tropical_horizontal_means.jld2 : as above, but the horizontal mean is
#                              restricted to the tropical band (|lat| <= 20°).
#
# These files are all `analyze_iteration` needs for the figures we plot.

using ClimaOcean
using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations: NORiBaseVerticalDiffusivity
using ConstructionBase: setproperties
using Oceananigans
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.Units
using Oceananigans.Fields: Field
using Oceananigans.Grids: φnode
using Oceananigans.AbstractOperations: Average
using Oceananigans.OutputWriters: JLD2Writer, AveragedTimeInterval
using Dates

# Tropical band (|latitude| <= 20°) condition for a horizontally-conditional
# `Average`. Signature matches Oceananigans' `condition` kernels:
# (i, j, k, grid, args...) -> Bool. Used so a horizontal mean restricted to
# the tropics can be written cheaply during the run (the global horizontal
# means carry no latitude axis, so a tropical profile can't be recovered
# post-hoc from them).
tropical_condition(i, j, k, grid, args...) =
    abs(φnode(i, j, k, grid, Center(), Center(), Center())) <= 20

# NORi calibrated defaults (from xkykai/NORiOceanParameterization.jl).
# Only the three parameters we expose for calibration are listed here.
const NORI_DEFAULTS = (
    νˢʰ       = 0.0615914063656973,
    Pr_shearₜ = 1.0842017486284887,
    Riᶜ       = 0.4366901962987793,
)

# NORi scaling spec: maps calibration parameter name → NORiBaseVerticalDiffusivity field.
const NORI_SCALING_SPEC = (
    νˢʰ_scaling       = :νˢʰ,
    Pr_shearₜ_scaling = :Pr_shearₜ,
    Riᶜ_scaling       = :Riᶜ,
)

"""
    build_nori_parameters(nori_scalings::AbstractDict)

Convert a dict of `<name>_scaling => value` into the NamedTuple of absolute
NORi parameter values, the keyword arguments of `NORiBaseVerticalDiffusivity`.
Each scaling multiplies its calibrated default (see `NORI_DEFAULTS`). Missing
scalings default to 1.0.
"""
function build_nori_parameters(nori_scalings::AbstractDict)
    params = Dict{Symbol,Float64}()
    for (param_name, field) in pairs(NORI_SCALING_SPEC)
        s = get(nori_scalings, String(param_name), 1.0)
        params[field] = NORI_DEFAULTS[field] * s
    end
    return (; (k => v for (k, v) in params)...)
end

# CATKE coefficients that one scaling multiplies together, which keeps their ratios
const coefficient_groups = Dict("Cᵘⁿ" => (:Cᵘⁿu, :Cᵘⁿc, :Cᵘⁿe, :CᵘⁿD),
                                "Cˡᵒʰⁱ" => (:Cˡᵒu, :Cˡᵒc, :Cˡᵒe, :CˡᵒD, :Cʰⁱu, :Cʰⁱc, :Cʰⁱe, :CʰⁱD),
                                "Cᶜ" => (:Cᶜu, :Cᶜc, :Cᶜe, :CᶜD))

"""
    scale_coefficients(coefficients, scalings)

`coefficients` with each property `name` multiplied by the `"<name>_scaling"` entry of `scalings` and by the
`"<group>_scaling"` entry of every group of `coefficient_groups` that holds `name`.
"""
function scale_coefficients(coefficients, scalings)
    scaling(name) = get(scalings, "$(name)_scaling", 1) *
                    prod(get(scalings, "$(group)_scaling", 1) for (group, names) in coefficient_groups if name in names; init = 1)
    scaled = (; (name => scaling(name) * getproperty(coefficients, name) for name in propertynames(coefficients))...)
    return setproperties(coefficients, scaled)
end

"""
    calibrated_catke(catke_scalings)

The CATKE closure of `OneDegree()` with its mixing-length and turbulent kinetic energy equation coefficients
multiplied by `catke_scalings`, a dict from `"<coefficient>_scaling"` to a scaling factor.
"""
function calibrated_catke(catke_scalings)
    catke = OneDegree().ocean.vertical_mixing
    return CATKEVerticalDiffusivity(time_discretization(catke);
                                    mixing_length = scale_coefficients(catke.mixing_length, catke_scalings),
                                    turbulent_kinetic_energy_equation = scale_coefficients(catke.turbulent_kinetic_energy_equation, catke_scalings),
                                    catke.maximum_tracer_diffusivity,
                                    catke.maximum_tke_diffusivity,
                                    catke.maximum_viscosity,
                                    catke.minimum_tke,
                                    catke.minimum_convective_buoyancy_flux,
                                    catke.negative_tke_damping_time_scale,
                                    catke.tke_time_step)
end

"""
    calibrated_gm(gm_scalings)

The Gent-McWilliams and Redi closure of `OneDegree()` with `κ_skew`, `κ_symmetric` and the `max_slope` of its flux
tapering multiplied by the `"κ_skew_scaling"`, `"κ_symmetric_scaling"` and `"max_slope_scaling"` entries of `gm_scalings`.
"""
function calibrated_gm(gm_scalings)
    gm = OneDegree().ocean.eddy_closure
    return IsopycnalSkewSymmetricDiffusivity(κ_skew = get(gm_scalings, "κ_skew_scaling", 1) * gm.κ_skew,
                                             κ_symmetric = get(gm_scalings, "κ_symmetric_scaling", 1) * gm.κ_symmetric,
                                             slope_limiter = scale_coefficients(gm.slope_limiter, gm_scalings))
end

"""
    attach_calibration_output_writers!(sim, output_dir, filename_prefix;
                                       stop_time, sampling_window, scalar_interval)

Attach the three calibration-only writers to a `Simulation` returned by
`omip_simulation`. The built-in OMIP diagnostics MUST already be disabled
(`diagnostics = false`) when constructing `sim` to avoid huge per-member
storage.
"""
function attach_calibration_output_writers!(sim, output_dir, filename_prefix;
                                            stop_time,
                                            sampling_window  = 5 * 365days,
                                            scalar_interval  = 30days)
    ocean = sim.model.ocean
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S
    u = ocean.model.velocities.u

    # 1. Single time mean over the final `sampling_window` of the run. The
    #    filename encodes the actual window length (in years) rather than a
    #    hardcoded "5", so a 3-year window writes `..._3year_average.jld2`.
    sampling_years = sampling_window / (365days)
    year_tag = isinteger(sampling_years) ? string(Int(sampling_years)) :
                                           string(round(sampling_years; digits = 2))
    ocean.output_writers[:calibration_final_mean] = JLD2Writer(
        ocean.model, (; T, S, u);
        schedule = AveragedTimeInterval(stop_time, window = sampling_window),
        filename = joinpath(output_dir, "$(filename_prefix)_$(year_tag)year_average.jld2"),
        overwrite_files = true,
    )

    # 2. Global-mean scalars vs time (for fig16 drift).
    T_bar = Field(Average(T))
    S_bar = Field(Average(S))
    ocean.output_writers[:calibration_global_means] = JLD2Writer(
        ocean.model, (T = T_bar, S = S_bar);
        schedule = AveragedTimeInterval(scalar_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_global_means.jld2"),
        overwrite_files = true,
    )

    # 3. Horizontal-mean 1-D profiles vs time (for fig17 / fig21).
    T_h = Field(Average(T; dims = (1, 2)))
    S_h = Field(Average(S; dims = (1, 2)))
    ocean.output_writers[:calibration_horizontal_means] = JLD2Writer(
        ocean.model, (T = T_h, S = S_h);
        schedule = AveragedTimeInterval(scalar_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_horizontal_means.jld2"),
        overwrite_files = true,
    )

    # 4. Tropical (|lat| <= 20°) horizontal-mean 1-D profiles vs time
    #    (for the tropics-only fig21 variants). Same as (3) but the
    #    horizontal average is conditioned to the tropical band.
    T_h_trop = Field(Average(T; dims = (1, 2), condition = tropical_condition))
    S_h_trop = Field(Average(S; dims = (1, 2), condition = tropical_condition))
    ocean.output_writers[:calibration_tropical_horizontal_means] = JLD2Writer(
        ocean.model, (T = T_h_trop, S = S_h_trop);
        schedule = AveragedTimeInterval(scalar_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_tropical_horizontal_means.jld2"),
        overwrite_files = true,
    )

    return sim
end

"""
    attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix;
                                       monthly_interval = 365days/12)

Attach a monthly-averaged 3-D T, S writer (`<prefix>_monthly_TS.jld2`) for the
seasonal-cycle calibration. Writes monthly-averaged 3-D T, S in a single file
over the whole run; the observation map reads the final 12 monthly T,S snapshots
(the last year's cycle), and the per-member video recomputes buoyancy from T,S.
Used in addition to `attach_calibration_output_writers!`.

Buoyancy is intentionally NOT written here: the OMIP ocean model carries
buoyancy === nothing, so Oceananigans.Models.buoyancy_operation(ocean.model)
returns a ZeroField. Buoyancy is instead recomputed from T,S downstream
(seasonal_buoyancy_3d), matching how the WOA reference is built.
"""
function attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix;
                                            monthly_interval = 365days / 12)
    ocean = sim.model.ocean
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S
    ocean.output_writers[:seasonal_monthly_TS] = JLD2Writer(
        ocean.model, (T = T, S = S);
        schedule = AveragedTimeInterval(monthly_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_monthly_TS.jld2"),
        overwrite_files = true,
    )
    return sim
end

"""
    attach_seasonal_monthly_EP_writer!(sim, output_dir, filename_prefix;
                                       monthly_interval = 365days/12)

Attach a monthly-averaged surface freshwater-flux writer (`<prefix>_monthly_EP.jld2`)
for the seasonal calibration's per-member E/P video: evaporation (water-vapor
mass flux), precipitation (prescribed total), and the net salinity flux. Same
field accessors as the OMIP `_monthly_surface` diagnostics. Visualization only —
not part of the calibration loss.
"""
function attach_seasonal_monthly_EP_writer!(sim, output_dir, filename_prefix;
                                            monthly_interval = 365days / 12)
    ocean = sim.model.ocean
    evap   = sim.model.interfaces.atmosphere_ocean_interface.fluxes.water_vapor
    # Total prescribed precipitation flux. Newer NumericalEarth splits the
    # combined condensate flux `Jᶜ` into rain (`Jʳⁿ`) and snow (`Jˢⁿ`); older
    # versions expose a single `Jᶜ`. Sum the components when both are present.
    atmos_state = sim.model.interfaces.exchanger.atmosphere.state
    precip = if hasproperty(atmos_state, :Jᶜ)
        atmos_state.Jᶜ
    else
        atmos_state.Jʳⁿ + atmos_state.Jˢⁿ
    end
    wfo    = sim.model.interfaces.net_fluxes.ocean.S
    ocean.output_writers[:seasonal_monthly_EP] = JLD2Writer(
        ocean.model, (evap = evap, precip = precip, wfo = wfo);
        schedule = AveragedTimeInterval(monthly_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_monthly_EP.jld2"),
        overwrite_files = true,
    )
    return sim
end

"""
    run_CATKE_GM_calibration_orca(catke_scalings, gm_scalings, config_dict)

Build and run a single ensemble member: `OneDegree()` with the closures of `calibrated_catke(catke_scalings)` and
`calibrated_gm(gm_scalings)`. `config_dict` must contain `"output_dir"` and may contain `"simulation_length"` (years,
default 10), `"sampling_length"` (years, default 5), `"forcing_dir"`, `"filename_prefix"`, `"use_gm"`, `"output_mode"`,
`"iteration"` and `"member"`.
"""
run_CATKE_GM_calibration_orca(catke_scalings::AbstractDict, gm_scalings::AbstractDict, config_dict::AbstractDict) =
    run_calibration_member(calibrated_catke(catke_scalings), gm_scalings, config_dict)

"""
    run_NORi_calibration_orca(nori_scalings, gm_scalings, config_dict)

As `run_CATKE_GM_calibration_orca`, with the NORi vertical diffusivity of `build_nori_parameters(nori_scalings)` and zero
background vertical diffusivity and viscosity.
"""
run_NORi_calibration_orca(nori_scalings::AbstractDict, gm_scalings::AbstractDict, config_dict::AbstractDict) =
    run_calibration_member(NORiBaseVerticalDiffusivity(; build_nori_parameters(nori_scalings)...), gm_scalings, config_dict;
                           background_diffusivity = 0, background_viscosity = 0)

function run_calibration_member(vertical_mixing, gm_scalings, config_dict; configuration_kw...)
    output_dir = config_dict["output_dir"]
    mkpath(output_dir)

    logfile_path     = joinpath(output_dir, "output.log")
    logfile          = open(logfile_path, "w")
    original_stdout  = stdout
    original_stderr  = stderr
    redirect_stdout(logfile)
    redirect_stderr(logfile)
    flusher = @async while isopen(logfile); flush(logfile); sleep(1); end

    try
        simulation_length = get(config_dict, "simulation_length", 10)   # years
        sampling_length   = get(config_dict, "sampling_length",   5)    # years
        forcing_dir       = get(config_dict, "forcing_dir", joinpath(homedir(), "JRA55_data"))
        filename_prefix   = get(config_dict, "filename_prefix", "orca_calib")
        output_mode       = Symbol(get(config_dict, "output_mode", "annual_mean"))
        iter              = get(config_dict, "iteration", -1)
        member            = get(config_dict, "member",    -1)

        eddy_closure = get(config_dict, "use_gm", true) ? calibrated_gm(gm_scalings) : nothing
        configuration = OneDegree(; vertical_mixing, eddy_closure, configuration_kw...)

        @info "Member $member, iter $iter: starting the calibration run"
        @info configuration
        @info "  vertical mixing = $vertical_mixing"
        @info "  simulation_length = $(simulation_length) years, sampling_length = $(sampling_length) years"
        @info "  output_dir = $output_dir"

        stop_time = simulation_length * 365days
        sim = omip_simulation(configuration; arch = GPU(), forcing_directory = forcing_dir, stop_time, diagnostics = false)

        attach_calibration_output_writers!(sim, output_dir, filename_prefix;
                                           stop_time,
                                           sampling_window = sampling_length * 365days)

        # Seasonal calibration also needs the monthly 3-D T,S (last-year cycle)
        # and monthly surface E/P for the per-member diagnostic videos.
        if output_mode === :seasonal
            attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix)
            attach_seasonal_monthly_EP_writer!(sim, output_dir, filename_prefix)
        end

        run!(sim)
        @info "Member $member, iter $iter: run! returned cleanly"
        return nothing
    catch e
        if e isa InterruptException
            println(stderr, "Interrupted by user")
        else
            println(stderr, "Error occurred: $e")
            rethrow(e)
        end
    finally
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
        close(logfile)
        println("Log file closed: $logfile_path")
    end
end

"""
    run_CATKE_GM_calibration_orca_dry_run(catke_scalings, gm_scalings, config_dict)

Cheap dry-run variant that does not actually launch the model — copies a
pre-existing 5-year-average JLD2 file from `\$HOME` into the member output
directory so the rest of the pipeline can be smoke-tested without paying
for GPU time.

Looks for `\$HOME/orca_calib_5year_average.jld2` as the source.
"""
function run_CATKE_GM_calibration_orca_dry_run(catke_scalings::AbstractDict,
                                               gm_scalings::AbstractDict,
                                               config_dict::AbstractDict)
    output_dir = config_dict["output_dir"]
    mkpath(output_dir)

    filename_prefix = get(config_dict, "filename_prefix", "orca_calib")
    src = joinpath(homedir(), "orca_calib_5year_average.jld2")
    dst = joinpath(output_dir, "$(filename_prefix)_5year_average.jld2")
    cp(src, dst; force = true)

    iter   = get(config_dict, "iteration", -1)
    member = get(config_dict, "member",    -1)
    @info "Dry run member=$member iter=$iter: copied $src → $dst"
    return nothing
end
