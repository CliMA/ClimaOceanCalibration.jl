# forward_model.jl
# Forward model for CATKE calibration
#
# Runs a 2-year ocean-sea ice simulation with CATKE closure using scaled parameters.
# Outputs monthly averages for the second year for comparison with ECCO observations.

# Check for UCX libraries that can cause issues with MPI+CUDA
ucx_libs = filter(lib -> occursin("ucx", lowercase(lib)), Libdl.dllist())
if isempty(ucx_libs)
    @info "✓ No UCX - safe to run!"
else
    @warn "✗ UCX libraries detected! This can cause issues with MPI+CUDA. Detected libs:\n$(join(ucx_libs, "\n"))"
end

"""
    run_CATKE_calibration_omip(Cˢ_scaling, Cᵘⁿ_scaling, Cᶜ_scaling, Cˢᵖ_scaling, Cᵉc_scaling, config_dict)

Run a 2-year OMIP simulation with scaled CATKE parameters.

# Arguments
- `Cˢ_scaling`: Scaling factor for surface layer TKE production (Cˢ)
- `Cᵘⁿ_scaling`: Scaling factor for unstable/convective mixing (Cᵘⁿu, Cᵘⁿc, Cᵘⁿe, CᵘⁿD)
- `Cᶜ_scaling`: Scaling factor for stable/convective mixing (Cᶜu, Cᶜc, Cᶜe, CᶜD)
- `Cˢᵖ_scaling`: Scaling factor for shear production (Cˢᵖ)
- `Cᵉc_scaling`: Scaling factor for TKE equation (Cᵉc)
- `config_dict`: Dictionary with configuration including:
  - `output_dir`: Directory to save output files

# Output
Saves monthly average files for year 2 (ocean_jan_average.jld2, etc.)
containing T, S, u, v, w, b, N² fields.
"""
function run_CATKE_calibration_omip(Cˢ_scaling, Cᵘⁿ_scaling, Cᶜ_scaling, Cˢᵖ_scaling, Cᵉc_scaling, config_dict)
    output_dir = config_dict["output_dir"]
    mkpath(output_dir)

    CATKE_default = ClimaOcean.Oceans.default_ocean_closure()

    Cˢ = CATKE_default.mixing_length.Cˢ * Cˢ_scaling

    Cᵘⁿu = CATKE_default.mixing_length.Cᵘⁿu * Cᵘⁿ_scaling
    Cᵘⁿc = CATKE_default.mixing_length.Cᵘⁿc * Cᵘⁿ_scaling
    Cᵘⁿe = CATKE_default.mixing_length.Cᵘⁿe * Cᵘⁿ_scaling
    CᵘⁿD = CATKE_default.turbulent_kinetic_energy_equation.CᵘⁿD * Cᵘⁿ_scaling

    Cᶜu = CATKE_default.mixing_length.Cᶜu * Cᶜ_scaling
    Cᶜc = CATKE_default.mixing_length.Cᶜc * Cᶜ_scaling
    Cᶜe = CATKE_default.mixing_length.Cᶜe * Cᶜ_scaling
    CᶜD = CATKE_default.turbulent_kinetic_energy_equation.CᶜD * Cᶜ_scaling

    Cᵉc = CATKE_default.mixing_length.Cᵉc * Cᵉc_scaling
    Cˢᵖ = CATKE_default.mixing_length.Cˢᵖ * Cˢᵖ_scaling

    mixing_length = CATKEMixingLength(; Cˢ, Cᵘⁿu, Cᵘⁿc, Cᵘⁿe, Cᶜu, Cᶜc, Cᶜe, Cᵉc, Cˢᵖ, Cᵇ=0.01)
    turbulent_kinetic_energy_equation = CATKEEquation(; Cᵂϵ=1.0, CᵘⁿD, CᶜD)
    catke_closure = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(); mixing_length, turbulent_kinetic_energy_equation)

    start_year = 1992
    simulation_length = 2

    arch = GPU()

    Nx = 720  # longitudinal direction
    Ny = 360  # meridional direction
    Nz = 100

    z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800)
    z_surf = z_faces(Nz)

    grid = TripolarGrid(arch;
                        size = (Nx, Ny, Nz),
                        z = z_faces,
                        halo = (7, 7, 7))

    bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

    tracer_advection   = WENO(order=7)
    momentum_advection = WENOVectorInvariant(order=5)
    free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=40minutes)

    @inline Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz) = 2 * (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))
    @inline geometric_νhb(i, j, k, grid, lx, ly, lz, clock, fields, λ) = Δ²ᵃᵃᵃ(i, j, k, grid, lx, ly, lz)^2 / λ

    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=geometric_νhb, discrete_form=true, parameters=25days)

    closure = (catke_closure, horizontal_viscosity)

    start_date = DateTime(start_year, 1, 1)
    end_date = start_date + Year(simulation_length)
    simulation_period = Dates.value(Second(end_date - start_date))

    ECCO_dir = joinpath(homedir(), "ECCO_data")
    mkpath(ECCO_dir)

    @info "Setting up salinity restoring..."
    @inline mask(x, y, z, t) = z >= z_surf - 1
    Smetadata = Metadata(:salinity; dataset=ECCO4Monthly(), dir=ECCO_dir, start_date, end_date)
    FS = DatasetRestoring(Smetadata, grid; rate = 1/30days, mask, time_indices_in_memory = 10)

    ocean = ocean_simulation(grid; Δt=1minutes,
                            momentum_advection,
                            tracer_advection,
                            timestepper = :SplitRungeKutta3,
                            free_surface,
                            forcing = (; S = FS),
                            closure)

    @info "Built ocean model $(ocean)"

    set!(ocean.model, T=Metadatum(:temperature; dataset=ECCO4Monthly(), date=start_date, dir=ECCO_dir),
                      S=Metadatum(:salinity;    dataset=ECCO4Monthly(), date=start_date, dir=ECCO_dir))
    @info "Initialized ocean fields with ECCO data"

    sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
    @info "Built sea ice model $(sea_ice)"

    set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), date=start_date, dir=ECCO_dir),
                        ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), date=start_date, dir=ECCO_dir))
    @info "Initialized sea ice fields with ECCO data"

    jra55_dir = joinpath(homedir(), "JRA55_data")
    mkpath(jra55_dir)
    dataset = MultiYearJRA55()
    backend = JRA55NetCDFBackend(100)

    @info "Setting up prescribed atmosphere $(dataset)"
    atmosphere = JRA55PrescribedAtmosphere(arch; dir=jra55_dir, dataset, backend, include_rivers_and_icebergs=true, start_date, end_date)
    radiation  = Radiation()

    @info "Built atmosphere model $(atmosphere)"

    omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

    @info "Built coupled model $(omip)"

    omip = Simulation(omip, Δt=30minutes, stop_time=simulation_period)
    @info "Built simulation $(omip)"

    u, v, w = ocean.model.velocities.u, ocean.model.velocities.v, ocean.model.velocities.w
    T, S = ocean.model.tracers.T, ocean.model.tracers.S
    b = buoyancy_field(ocean.model)
    N² = Field(buoyancy_frequency(ocean.model))
    h, ℵ = sea_ice.model.ice_thickness, sea_ice.model.ice_concentration

    ocean_outputs = (; u, v, w, T, S, b, N²)
    sea_ice_outputs = (; h, ℵ)

    final_year_jan = end_date - Year(1)
    final_year_months = final_year_jan:Month(1):end_date
    final_year_month_lengths = Dates.value.(Dates.Second.(diff(final_year_months)))
    month_names = [:jan, :feb, :mar, :apr, :may, :jun, :jul, :aug, :sep, :oct, :nov, :dec]

    for (i, (month_name, month_length)) in enumerate(zip(month_names, final_year_month_lengths))
        month_start = final_year_months[i]
        month_end = final_year_months[i+1]

        ocean.output_writers[Symbol("$(month_name)_average")] = JLD2Writer(ocean.model, ocean_outputs;
                                                    schedule = AveragedTimeInterval(Dates.value(Dates.Second(month_end - start_date)),
                                                                window=month_length),
                                                    filename = "$(output_dir)/ocean_$(month_name)_average",
                                                    overwrite_existing = true)

        sea_ice.output_writers[Symbol("$(month_name)_average")] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                    schedule = AveragedTimeInterval(Dates.value(Dates.Second(month_end - start_date)),
                                                                window=month_length),
                                                    filename = "$(output_dir)/sea_ice_$(month_name)_average",
                                                    overwrite_existing = true)
    end

    wall_time = Ref(time_ns())

    function progress(sim)
        sea_ice = sim.model.sea_ice
        ocean   = sim.model.ocean
        hmax = maximum(sea_ice.model.ice_thickness)
        ℵmax = maximum(sea_ice.model.ice_concentration)
        Tmax = maximum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        Tmin = minimum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
        umax = maximum(ocean.model.velocities.u)
        vmax = maximum(ocean.model.velocities.v)
        wmax = maximum(ocean.model.velocities.w)

        step_time = 1e-9 * (time_ns() - wall_time[])

        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
        msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
        msg4 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
        msg5 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m/s, ", umax, vmax, wmax)
        msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

        @info msg1 * msg2 * msg4 * msg5 * msg6

        wall_time[] = time_ns()

        return nothing
    end

    add_callback!(omip, progress, IterationInterval(100))

    run!(omip)
end
