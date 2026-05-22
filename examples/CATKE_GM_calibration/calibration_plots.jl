# calibration_plots.jl
# Per-member diagnostic figures rendered from each iteration's outputs.
#
# Each member directory carries three calibration files:
#   <prefix>_5year_average.jld2      — final 5-year mean of T, S, u
#   <prefix>_global_means.jld2       — 30-day-averaged global mean T, S vs time
#   <prefix>_horizontal_means.jld2   — 30-day-averaged horizontal mean T(z), S(z) vs time
#
# That's enough to plot drift time series (fig16-style), final horizontal-mean
# profiles vs WOA (fig17-style), the time×depth drift heatmap (fig21-style),
# and a tropical T/S vertical-mean bias map (the calibration target itself).
#
# Zonal-mean sections (fig18/19) and equatorial undercurrent (fig25) require
# a ConservativeRegridding regridder on the ORCA grid, which is too expensive
# to build inside the calibration loop. Run examples/OMIP_GCP/visualize_omip.jl
# against a member's 5year_average.jld2 to render those post-hoc.

using CairoMakie
using JLD2
using Oceananigans
using Oceananigans.Grids: λnodes, φnodes, znodes
using Oceananigans.Fields: Field, location, interior
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Statistics
using NaNStatistics

# Bring in the same load helpers used by observation_map so plots and the
# loss function read the model state through the same code path.
include(joinpath(@__DIR__, "data_processing.jl"))

const SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Compute horizontal mean of an Oceananigans Field on the ORCA grid using
# straight nanmean over (i, j). Approximate (cells aren't equal-area), but
# good enough for a calibration diagnostic.
function _horizontal_nanmean(field::Field)
    f = on_architecture(CPU(), field)
    mask_immersed_field!(f, NaN)
    data = interior(f)
    return [nanmean(@view data[:, :, k]) for k in 1:size(data, 3)]
end

function _z_centers(field::Field)
    LX, LY, LZ = location(field)
    return znodes(field.grid, LX(), LY(), LZ())
end

"""
    plot_drift_timeseries(member_dir, filename_prefix, out_path)

fig16-style: global-mean T,S drift vs time.
"""
function plot_drift_timeseries(member_dir, filename_prefix, out_path)
    path = joinpath(member_dir, "$(filename_prefix)_global_means.jld2")
    isfile(path) || (@warn "missing $path"; return)

    T_fts = FieldTimeSeries(path, "T")
    S_fts = FieldTimeSeries(path, "S")
    t_years = T_fts.times ./ SECONDS_PER_YEAR
    T = [first(interior(T_fts[i])) for i in 1:length(T_fts.times)]
    S = [first(interior(S_fts[i])) for i in 1:length(S_fts.times)]

    fig = Figure(size = (1000, 400), fontsize = 14)
    axT = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "ΔT (°C)",
               title = "Global-mean temperature drift")
    lines!(axT, t_years, T .- T[1]; linewidth = 2)
    axS = Axis(fig[1, 2]; xlabel = "Time (years)", ylabel = "ΔS (PSU)",
               title = "Global-mean salinity drift")
    lines!(axS, t_years, S .- S[1]; linewidth = 2)
    save(out_path, fig)
    return out_path
end

"""
    plot_horizontal_mean_profile(member_dir, filename_prefix, woa_T, woa_S, sampling_length, out_path)

fig17-style: horizontal-mean T,S profile (time-mean of the last
`sampling_length` years of the horizontal_means file) compared to WOA's
horizontal-mean profile.
"""
function plot_horizontal_mean_profile(member_dir, filename_prefix, woa_T::Field, woa_S::Field,
                                      sampling_length, out_path)
    path = joinpath(member_dir, "$(filename_prefix)_horizontal_means.jld2")
    isfile(path) || (@warn "missing $path"; return)

    T_fts = FieldTimeSeries(path, "T")
    S_fts = FieldTimeSeries(path, "S")
    times = T_fts.times
    end_t = last(times)
    sample_t = end_t - sampling_length * SECONDS_PER_YEAR
    idx = findall(t -> t >= sample_t, times)
    T_prof = mean(hcat([vec(interior(T_fts[i])) for i in idx]...); dims = 2) |> vec
    S_prof = mean(hcat([vec(interior(S_fts[i])) for i in idx]...); dims = 2) |> vec
    z = _z_centers(T_fts[1])

    T_woa = _horizontal_nanmean(woa_T)
    S_woa = _horizontal_nanmean(woa_S)
    z_woa = _z_centers(woa_T)

    fig = Figure(size = (1000, 600), fontsize = 14)
    axT = Axis(fig[1, 1]; xlabel = "Temperature (°C)", ylabel = "Depth (m)",
               title = "Horizontal-mean temperature (last $sampling_length yr)")
    lines!(axT, T_prof, z; linewidth = 2, label = "Model")
    lines!(axT, T_woa, z_woa; linewidth = 2, linestyle = :dash, label = "WOA")
    ylims!(axT, (-5500, 0)); axislegend(axT; position = :rb)

    axS = Axis(fig[1, 2]; xlabel = "Salinity (PSU)", ylabel = "Depth (m)",
               title = "Horizontal-mean salinity (last $sampling_length yr)")
    lines!(axS, S_prof, z; linewidth = 2, label = "Model")
    lines!(axS, S_woa, z_woa; linewidth = 2, linestyle = :dash, label = "WOA")
    ylims!(axS, (-5500, 0)); axislegend(axS; position = :rb)

    save(out_path, fig)
    return out_path
end

"""
    plot_TS_drift_heatmap(member_dir, filename_prefix, out_path)

fig21-style: horizontal-mean T,S drift relative to the first record, as
a time × depth heatmap.
"""
function plot_TS_drift_heatmap(member_dir, filename_prefix, out_path)
    path = joinpath(member_dir, "$(filename_prefix)_horizontal_means.jld2")
    isfile(path) || (@warn "missing $path"; return)

    T_fts = FieldTimeSeries(path, "T")
    S_fts = FieldTimeSeries(path, "S")
    t = T_fts.times ./ SECONDS_PER_YEAR
    z = _z_centers(T_fts[1])

    Nt = length(t)
    T_mat = hcat([vec(interior(T_fts[i])) for i in 1:Nt]...) # z × t
    S_mat = hcat([vec(interior(S_fts[i])) for i in 1:Nt]...)
    ΔT = T_mat .- T_mat[:, 1]
    ΔS = S_mat .- S_mat[:, 1]
    Tmax = max(maximum(abs, filter(isfinite, ΔT)), 1e-6)
    Smax = max(maximum(abs, filter(isfinite, ΔS)), 1e-6)

    fig = Figure(size = (1100, 700), fontsize = 14)
    axT = Axis(fig[1, 1]; xlabel = "Time (years)", ylabel = "Depth (m)",
               title = "ΔT (°C)")
    hmT = heatmap!(axT, t, z, transpose(ΔT);
                   colormap = :balance, colorrange = (-Tmax, Tmax))
    Colorbar(fig[1, 2], hmT; label = "°C")
    ylims!(axT, (-5500, 0))

    axS = Axis(fig[2, 1]; xlabel = "Time (years)", ylabel = "Depth (m)",
               title = "ΔS (PSU)")
    hmS = heatmap!(axS, t, z, transpose(ΔS);
                   colormap = :balance, colorrange = (-Smax, Smax))
    Colorbar(fig[2, 2], hmS; label = "PSU")
    ylims!(axS, (-5500, 0))

    save(out_path, fig)
    return out_path
end

"""
    plot_tropical_target_bias(member_dir, filename_prefix, woa_T, woa_S, lat_range, z_min, out_path)

Plot the calibration target itself: 5-year-mean T,S bias vs WOA, vertically
averaged over the upper |z_min| m, restricted to the tropics. Lat-lon
heatmap of (T_model - T_woa) and (S_model - S_woa).
"""
function plot_tropical_target_bias(member_dir, filename_prefix,
                                   woa_T::Field, woa_S::Field,
                                   lat_range, z_min, out_path)
    T, S, _ = load_orca_5yr_average(member_dir, filename_prefix)

    function vmean_top(field::Field, ref::Field)
        f = on_architecture(CPU(), field)
        r = on_architecture(CPU(), ref)
        LX, LY, LZ = location(f)
        z = znodes(f.grid, LX(), LY(), LZ())
        kkeep = findall(zᵢ -> zᵢ >= z_min, z)
        mask_immersed_field!(f, NaN)
        mask_immersed_field!(r, NaN)
        fd = interior(f, :, :, kkeep)
        rd = interior(r, :, :, kkeep)
        return dropdims(nanmean(fd; dims = 3); dims = 3),
               dropdims(nanmean(rd; dims = 3); dims = 3),
               φnodes(f.grid, LX(), LY(), LZ()),
               λnodes(f.grid, LX(), LY(), LZ())
    end

    Tm, Tw, φ, λ = vmean_top(T, woa_T)
    Sm, Sw, _, _ = vmean_top(S, woa_S)
    jkeep = findall(φᵢ -> lat_range[1] <= φᵢ <= lat_range[2], φ)

    Tbias = (Tm .- Tw)[:, jkeep]
    Sbias = (Sm .- Sw)[:, jkeep]
    φsub  = φ[jkeep]

    Tmax = max(maximum(abs, filter(isfinite, Tbias)), 1e-6)
    Smax = max(maximum(abs, filter(isfinite, Sbias)), 1e-6)

    fig = Figure(size = (1300, 600), fontsize = 14)
    axT = Axis(fig[1, 1]; xlabel = "Longitude index", ylabel = "Latitude",
               title = "T bias (model − WOA), upper $(abs(z_min)) m")
    hmT = heatmap!(axT, 1:size(Tbias, 1), φsub, Tbias;
                   colormap = :balance, colorrange = (-Tmax, Tmax),
                   nan_color = :lightgray)
    Colorbar(fig[1, 2], hmT; label = "°C")

    axS = Axis(fig[1, 3]; xlabel = "Longitude index", ylabel = "Latitude",
               title = "S bias (model − WOA), upper $(abs(z_min)) m")
    hmS = heatmap!(axS, 1:size(Sbias, 1), φsub, Sbias;
                   colormap = :balance, colorrange = (-Smax, Smax),
                   nan_color = :lightgray)
    Colorbar(fig[1, 4], hmS; label = "PSU")

    save(out_path, fig)
    return out_path
end

"""
    plot_member_vs_woa(member_dir, filename_prefix, woa_file, lat_range, z_min, sampling_length, fig_dir)

Render the full per-member diagnostic figure set into `fig_dir`.
"""
function plot_member_vs_woa(member_dir::AbstractString,
                            filename_prefix::AbstractString,
                            woa_file::AbstractString,
                            lat_range::Tuple,
                            z_min::Real,
                            sampling_length::Real,
                            fig_dir::AbstractString)
    mkpath(fig_dir)
    woa_T, woa_S = load_woa_on_orca(woa_file)

    try
        plot_drift_timeseries(member_dir, filename_prefix,
                              joinpath(fig_dir, "fig16_drift.png"))
    catch e
        @warn "plot_drift_timeseries failed" exception=e
    end

    try
        plot_horizontal_mean_profile(member_dir, filename_prefix, woa_T, woa_S,
                                     sampling_length,
                                     joinpath(fig_dir, "fig17_profiles.png"))
    catch e
        @warn "plot_horizontal_mean_profile failed" exception=e
    end

    try
        plot_TS_drift_heatmap(member_dir, filename_prefix,
                              joinpath(fig_dir, "fig21_TS_drift_heatmap.png"))
    catch e
        @warn "plot_TS_drift_heatmap failed" exception=e
    end

    try
        plot_tropical_target_bias(member_dir, filename_prefix, woa_T, woa_S,
                                  lat_range, z_min,
                                  joinpath(fig_dir, "tropical_target_bias.png"))
    catch e
        @warn "plot_tropical_target_bias failed" exception=e
    end

    return fig_dir
end
