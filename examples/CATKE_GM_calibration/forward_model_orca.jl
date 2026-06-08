# forward_model_orca.jl
# Per-member forward model: runs omip_simulation(:orca; ...) with CATKE/GM
# scalings applied multiplicatively to the upstream defaults, with the
# built-in diagnostics disabled and a small set of custom output writers
# attached:
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

using ClimaOceanCalibration
using ClimaOceanCalibration.OMIPSimulations
using Oceananigans
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

# Default CATKE / GM physics. Defaults match Oceananigans `CATKEMixingLength` /
# `CATKEEquation` (which is what `omip_simulation` builds from when no override
# is passed). The only explicit override in `omip_closure` is `Cᵇ = 0.28`, which
# already matches the Oceananigans default. GM defaults set to (1000, 1000) per
# the orca calibration target run.
const CATKE_ML_DEFAULTS = (
    Cˢ   = 1.131,
    Cᵇ   = 0.28,
    Cˢᵖ  = 0.505,
    CRiᵟ = 1.02,
    CRi⁰ = 0.254,
    Cʰⁱu = 0.242, Cˡᵒu = 0.361, Cᵘⁿu = 0.370, Cᶜu = 3.705,
    Cʰⁱc = 0.098, Cˡᵒc = 0.369, Cᵘⁿc = 0.572, Cᶜc = 4.793, Cᵉc = 0.112,
    Cʰⁱe = 0.548, Cˡᵒe = 7.863, Cᵘⁿe = 1.447, Cᶜe = 3.642,
    # Note: Cᵉu = Cᵉe = 0.0 by default; scaling is degenerate so omitted.
)
const CATKE_TKE_DEFAULTS = (
    CʰⁱD = 0.579, CˡᵒD = 1.604, CᵘⁿD = 0.923, CᶜD = 3.254,
    Cᵂu★ = 3.179, CᵂwΔ = 0.383, Cᵂϵ  = 1.0,
    # Note: CᵉD = 0.0 by default; scaling is degenerate so omitted.
)
const GM_DEFAULTS = (
    κ_skew                  = 1000.0,
    κ_symmetric             = 1000.0,
    slope_limiter_max_slope = 1e-2,   # FluxTapering default
)

# CATKE scaling parameter spec: maps each *calibration parameter* name to a
# list of (sub-struct, fieldname) pairs that all share the same scaling
# factor. sub-struct is :ml (CATKEMixingLength) or :tke (CATKEEquation).
# One scaling per coefficient — fine-grained. Drop unwanted ones in the
# top-level calibration script by removing their names from
# `catke_param_names`.
const CATKE_SCALING_SPEC = (
    # CATKEMixingLength
    Cˢ_scaling   = ((:ml, :Cˢ),),
    Cᵇ_scaling   = ((:ml, :Cᵇ),),
    Cˢᵖ_scaling  = ((:ml, :Cˢᵖ),),
    CRiᵟ_scaling = ((:ml, :CRiᵟ),),
    CRi⁰_scaling = ((:ml, :CRi⁰),),
    Cʰⁱu_scaling = ((:ml, :Cʰⁱu),),
    Cˡᵒu_scaling = ((:ml, :Cˡᵒu),),
    Cᵘⁿu_scaling = ((:ml, :Cᵘⁿu),),
    Cᶜu_scaling  = ((:ml, :Cᶜu),),
    Cʰⁱc_scaling = ((:ml, :Cʰⁱc),),
    Cˡᵒc_scaling = ((:ml, :Cˡᵒc),),
    Cᵘⁿc_scaling = ((:ml, :Cᵘⁿc),),
    Cᶜc_scaling  = ((:ml, :Cᶜc),),
    Cᵉc_scaling  = ((:ml, :Cᵉc),),
    Cʰⁱe_scaling = ((:ml, :Cʰⁱe),),
    Cˡᵒe_scaling = ((:ml, :Cˡᵒe),),
    Cᵘⁿe_scaling = ((:ml, :Cᵘⁿe),),
    Cᶜe_scaling  = ((:ml, :Cᶜe),),
    # CATKEEquation
    CʰⁱD_scaling = ((:tke, :CʰⁱD),),
    CˡᵒD_scaling = ((:tke, :CˡᵒD),),
    CᵘⁿD_scaling = ((:tke, :CᵘⁿD),),
    CᶜD_scaling  = ((:tke, :CᶜD),),
    Cᵂu★_scaling = ((:tke, :Cᵂu★),),
    CᵂwΔ_scaling = ((:tke, :CᵂwΔ),),
    Cᵂϵ_scaling  = ((:tke, :Cᵂϵ),),
)
# Each entry: (location, field). location ∈ {:top, :slope_limiter}. :top
# entries land directly on `gm_parameters`; :slope_limiter entries go in
# the nested `slope_limiter = (; …)` NamedTuple consumed by
# `_with_nested_constructor_overrides` in omip_closure.
const GM_SCALING_SPEC = (
    κ_skew_scaling      = (:top,           :κ_skew),
    κ_symmetric_scaling = (:top,           :κ_symmetric),
    max_slope_scaling   = (:slope_limiter, :max_slope),
)

"""
    build_catke_parameters(catke_scalings::Dict{String,<:Real})

Convert a dict of `<name>_scaling => value` into the structured
`catke_parameters` NamedTuple expected by `omip_simulation`.

Each scaling multiplies its underlying default (see `CATKE_ML_DEFAULTS` /
`CATKE_TKE_DEFAULTS`). Scalings that are not in the dict are treated as 1
(i.e., the upstream default is used unchanged).
"""
function build_catke_parameters(catke_scalings::AbstractDict)
    ml  = Dict{Symbol,Float64}()
    tke = Dict{Symbol,Float64}()
    for (param_name, mapping) in pairs(CATKE_SCALING_SPEC)
        s = get(catke_scalings, String(param_name), 1.0)
        for (loc, field) in mapping
            if loc === :ml
                ml[field]  = CATKE_ML_DEFAULTS[field]  * s
            elseif loc === :tke
                tke[field] = CATKE_TKE_DEFAULTS[field] * s
            else
                error("Unknown CATKE sub-struct: $loc")
            end
        end
    end
    ml_nt  = (; (Symbol(k) => v for (k, v) in ml)...)
    tke_nt = (; (Symbol(k) => v for (k, v) in tke)...)
    return (; mixing_length = ml_nt, tke_equation = tke_nt)
end

"""
    build_gm_parameters(gm_scalings::Dict{String,<:Real})

Convert a dict of `<name>_scaling => value` into the structured
`gm_parameters` NamedTuple expected by `omip_simulation`. Top-level
entries (`κ_skew`, `κ_symmetric`) land directly on the NamedTuple;
`slope_limiter` entries are emitted as a nested
`(; max_slope = …)` NamedTuple. Missing scalings default to 1.0.
"""
function build_gm_parameters(gm_scalings::AbstractDict)
    top            = Dict{Symbol,Float64}()
    slope_limiter  = Dict{Symbol,Float64}()
    for (param_name, (loc, field)) in pairs(GM_SCALING_SPEC)
        s = get(gm_scalings, String(param_name), 1.0)
        default = if loc === :top
            GM_DEFAULTS[field]
        elseif loc === :slope_limiter
            GM_DEFAULTS[Symbol(:slope_limiter_, field)]
        else
            error("Unknown GM scaling location: $loc")
        end
        scaled = default * s
        if loc === :top
            top[field] = scaled
        else  # :slope_limiter
            slope_limiter[field] = scaled
        end
    end
    top_nt = (; (k => v for (k, v) in top)...)
    if isempty(slope_limiter)
        return top_nt
    end
    sl_nt = (; (k => v for (k, v) in slope_limiter)...)
    return merge(top_nt, (; slope_limiter = sl_nt))
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
        overwrite_existing = true,
    )

    # 2. Global-mean scalars vs time (for fig16 drift).
    T_bar = Field(Average(T))
    S_bar = Field(Average(S))
    ocean.output_writers[:calibration_global_means] = JLD2Writer(
        ocean.model, (T = T_bar, S = S_bar);
        schedule = AveragedTimeInterval(scalar_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_global_means.jld2"),
        overwrite_existing = true,
    )

    # 3. Horizontal-mean 1-D profiles vs time (for fig17 / fig21).
    T_h = Field(Average(T; dims = (1, 2)))
    S_h = Field(Average(S; dims = (1, 2)))
    ocean.output_writers[:calibration_horizontal_means] = JLD2Writer(
        ocean.model, (T = T_h, S = S_h);
        schedule = AveragedTimeInterval(scalar_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_horizontal_means.jld2"),
        overwrite_existing = true,
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
        overwrite_existing = true,
    )

    return sim
end

"""
    attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix;
                                       monthly_interval = 365days/12)

Attach a monthly-averaged 3-D T, S writer (`<prefix>_monthly_TS.jld2`) for the
seasonal-cycle calibration. Writes monthly-averaged 3-D T, S, and buoyancy in a
single file over the whole run; the observation map reads the final 12 monthly
T,S snapshots (the last year's cycle) and the per-member video also reads
buoyancy. Used in addition to `attach_calibration_output_writers!`.
"""
function attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix;
                                            monthly_interval = 365days / 12)
    ocean = sim.model.ocean
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S
    bo = Oceananigans.Models.buoyancy_operation(ocean.model)
    ocean.output_writers[:seasonal_monthly_TS] = JLD2Writer(
        ocean.model, (T = T, S = S, bo = bo);
        schedule = AveragedTimeInterval(monthly_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_monthly_TS.jld2"),
        overwrite_existing = true,
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
    precip = sim.model.interfaces.exchanger.atmosphere.state.Jᶜ
    wfo    = sim.model.interfaces.net_fluxes.ocean.S
    ocean.output_writers[:seasonal_monthly_EP] = JLD2Writer(
        ocean.model, (evap = evap, precip = precip, wfo = wfo);
        schedule = AveragedTimeInterval(monthly_interval),
        filename = joinpath(output_dir, "$(filename_prefix)_monthly_EP.jld2"),
        overwrite_existing = true,
    )
    return sim
end

"""
    run_CATKE_GM_calibration_orca(catke_scalings, gm_scalings, config_dict)

Build and run a single ensemble member's ORCA simulation.

`catke_scalings` and `gm_scalings` are dicts from `<name>_scaling` to a
scaling factor. `config_dict` must contain `"output_dir"` and may
optionally contain `"simulation_length"` (years, default 10),
`"sampling_length"` (years, default 5), `"forcing_dir"`, `"restoring_dir"`,
`"staging_dir"`, `"iteration"`, `"member"`.
"""
function run_CATKE_GM_calibration_orca(catke_scalings::AbstractDict,
                                       gm_scalings::AbstractDict,
                                       config_dict::AbstractDict)
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
        forcing_dir       = get(config_dict, "forcing_dir",
                                joinpath(homedir(), "JRA55_data"))
        restoring_dir     = get(config_dict, "restoring_dir",
                                joinpath(homedir(), "ECCO_data"))
        staging_dir       = get(config_dict, "staging_dir",
                                joinpath(output_dir, "staged_data"))
        filename_prefix   = get(config_dict, "filename_prefix", "orca_calib")
        iter              = get(config_dict, "iteration", -1)
        member            = get(config_dict, "member",    -1)

        catke_parameters = build_catke_parameters(catke_scalings)
        gm_parameters    = build_gm_parameters(gm_scalings)

        use_gm = get(config_dict, "use_gm", true)
        if !use_gm
            # Sentinel in omip_closure: κ_skew or κ_symmetric == 0 disables the
            # IsopycnalSkewSymmetricDiffusivity entirely.
            gm_parameters = (; κ_skew = 0, κ_symmetric = 0)
        end

        with_ice_dynamics = get(config_dict, "with_ice_dynamics", true)
        Δz_top            = get(config_dict, "Δz_top", nothing)
        skin_temperature  = get(config_dict, "skin_temperature", false)
        # Initial climatology + output mode. Defaults preserve the legacy
        # annual-mean calibration; the seasonal calibration passes
        # initial_field=:monthly and output_mode=:seasonal.
        initial_field     = Symbol(get(config_dict, "initial_field", "annual"))
        output_mode       = Symbol(get(config_dict, "output_mode",   "annual_mean"))

        @info "Member $member, iter $iter: starting ORCA calibration run"
        @info "  use_gm            = $use_gm"
        @info "  with_ice_dynamics = $with_ice_dynamics"
        @info "  Δz_top            = $(Δz_top === nothing ? "default" : Δz_top)"
        @info "  skin_temperature  = $skin_temperature"
        @info "  catke_parameters = $catke_parameters"
        @info "  gm_parameters    = $gm_parameters"
        @info "  simulation_length = $(simulation_length) years, sampling_length = $(sampling_length) years"
        @info "  output_dir = $output_dir"

        stop_time = simulation_length * 365days

        sim = omip_simulation(:orca;
                              arch  = GPU(),
                              Nz    = 70,
                              depth = 5500,
                              Δz_top,
                              catke_parameters,
                              gm_parameters,
                              biharmonic_timescale = 50days,
                              flux_configuration   = :corrected,
                              with_snow            = true,
                              skin_temperature,
                              with_ice_dynamics,
                              initial_field,
                              diagnostics          = false,
                              Δt              = 30minutes,
                              forcing_dir,
                              restoring_dir,
                              staging_dir,
                              output_dir,
                              filename_prefix)

        attach_calibration_output_writers!(sim, output_dir, filename_prefix;
                                           stop_time,
                                           sampling_window = sampling_length * 365days)

        # Seasonal calibration also needs the monthly 3-D T,S,b (last-year cycle)
        # and monthly surface E/P for the per-member diagnostic videos.
        if output_mode === :seasonal
            attach_seasonal_monthly_TS_writer!(sim, output_dir, filename_prefix)
            attach_seasonal_monthly_EP_writer!(sim, output_dir, filename_prefix)
        end

        sim.stop_time = stop_time
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
