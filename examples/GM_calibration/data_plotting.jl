using CairoMakie
using Oceananigans
using Oceananigans.Grids: znodes, φnodes
using NaNStatistics
using ColorSchemes

function plot_parameter_distribution(κs, error)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="κ skew (m²/s)", ylabel="κ symmetric (m²/s)", title="Parameter Distribution, Covariance-weighted loss at mean = $error")
    scatter!(ax, κs[1, :], κs[2, :])
    return fig
end

function exp_levels(min_val, max_val, n; factor=2)
    uniform = range(0, 1, length=n)
    transformed = (exp.(factor .* uniform) .- 1) ./ (exp(factor) - 1)
    return min_val .+ transformed .* (max_val - min_val)
end

function inverted_exp_levels(min_val, max_val, n; factor=2)
    uniform = range(0, 1, length=n)
    # This inverts the concentration from low to high values
    transformed = 1 .- (exp.(factor .* (1 .- uniform)) .- 1) ./ (exp(factor) - 1)
    return min_val .+ transformed .* (max_val - min_val)
end

function plot_zonal_average(truth_data, model_data, field_name, κ_skew, κ_symmetric)
    fig = Figure(size=(1920, 1080))
    axtruth = Axis(fig[1, 1]; xlabel="Latitude (°)", ylabel="Depth (m)", title="Target: ECCO4")
    axmodel = Axis(fig[2, 1]; xlabel="Latitude (°)", ylabel="Depth (m)", title="Model Output")
    axdiff = Axis(fig[3, 1]; xlabel="Latitude (°)", ylabel="Depth (m)", title="Anomaly (Model - Target)")

    LX, LY, LZ = location(truth_data)
    zCs = znodes(truth_data.grid, LX(), LY(), LZ())

    truth_φCs = φnodes(truth_data.grid, LX(), LY(), LZ())
    model_φCs = φnodes(model_data.grid, LX(), LY(), LZ())

    truth_field = nanmean(interior(truth_data), dims=1)[1, :, :]
    model_field = nanmean(interior(model_data), dims=1)[1, :, :]
    diff_field = model_field .- truth_field

    fieldlim = (nanminimum([truth_field; model_field]), nanmaximum([truth_field; model_field]))
    difflim = (-nanmaximum(abs.(diff_field)) / 2, nanmaximum(abs.(diff_field)) / 2)

    if field_name == "T"
        @info "Plotting temperature zonal average..."
        field_levels = exp_levels(fieldlim[1], fieldlim[2], 15)
        field_colorbar_kwargs = (label="Temperature (°C)",)
        diff_colorbar_kwargs = (label="Temperature Anomaly (°C)",)
        difflim = (-0.7, 0.7)
    elseif field_name == "S"
        @info "Plotting salinity zonal average..."
        field_colorbar_kwargs = (label="Salinity (psu)",)
        diff_colorbar_kwargs = (label="Salinity Anomaly (psu)",)
        # fieldlim = (nanminimum([truth_field; model_field[2:end, :]]), nanmaximum([truth_field; model_field[2:end, :]]))
        # difflim = (-nanmaximum(abs.(diff_field[2:end, :])) / 2, nanmaximum(abs.(diff_field[2:end, :])) / 2)
        fieldlim = (34, 36)
        difflim = (-0.1, 0.1)
        # field_levels = inverted_exp_levels(fieldlim[1], fieldlim[2], 15)
        # field_levels = range(fieldlim[1], fieldlim[2], length=15)
        field_levels = exp_levels(fieldlim[1], fieldlim[2], 15)
    elseif field_name == "b"
        @info "Plotting buoyancy zonal average..."
        field_levels = exp_levels(fieldlim[1], fieldlim[2], 15)
        difflim = (-0.001, 0.001)
        field_colorbar_kwargs = (label="Buoyancy (m/s²)",)
        diff_colorbar_kwargs = (label="Buoyancy Anomaly (m/s²)",)
    else
        error("Unsupported field name: $field_name")
    end

    diff_levels = range(difflim[1], difflim[2], length=15)

    cf_f = contourf!(axtruth, truth_φCs, zCs, truth_field, colormap=:turbo, levels = field_levels, extendhigh=:auto, extendlow=:auto)
    contourf!(axmodel, model_φCs, zCs, model_field, colormap=:turbo, levels = field_levels, extendhigh=:auto, extendlow=:auto)
    cf_d = contourf!(axdiff, model_φCs, zCs, diff_field, colormap=:balance, levels = diff_levels, extendhigh=:auto, extendlow=:auto)

    Colorbar(fig[1:2, 2], cf_f; field_colorbar_kwargs...)
    Colorbar(fig[3, 2], cf_d; diff_colorbar_kwargs...)

    xlims!(axtruth, -84, 84)
    xlims!(axmodel, -84, 84)
    xlims!(axdiff, -84, 84)
    ylims!(axtruth, -6000, 0)
    ylims!(axmodel, -6000, 0)
    ylims!(axdiff, -6000, 0)

    Label(fig[0, 1:2], "Zonal Average of $field_name (κ_skew=$(round(κ_skew, digits=1)), κ_symmetric=$(round(κ_symmetric)))", fontsize=25, font=:bold)
    return fig
end