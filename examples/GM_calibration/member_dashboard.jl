# Interactive dashboard for one half-degree GM calibration member:
#
#     julia --project examples/GM_calibration/member_dashboard.jl calibration_runs/<case>/iteration_000/member_001
#
# The member's files carry a grid the current Oceananigans cannot deserialize, so the tripolar
# grid of `half_degree_omip.jl` is rebuilt here with the bottom under the deepest wet cell.

isempty(ARGS) && error("usage: julia --project examples/GM_calibration/member_dashboard.jl member_dir")

using GLMakie
using Dates
using JLD2
using Oceananigans
using ClimaOceanCalibration.Visualization

dir = first(ARGS)
Nx, Ny, Nz, H = 720, 360, 100, 7
underlying_grid = TripolarGrid(CPU(); size = (Nx, Ny, Nz), z = ExponentialDiscretization(Nz, -6000, 0; scale = 1800), halo = (H, H, H))

S = jldopen(joinpath(dir, "ocean_complete_fields_10year_average_calibrationsample.jld2")) do file
    iteration = first(filter(!=("serialized"), keys(file["timeseries/S"])))
    file["timeseries/S/$iteration"][H+1:end-H, H+1:end-H, H+1:end-H]
end
zᶠ = znodes(underlying_grid, Face())
bottom_height = [(k = findfirst(!=(0), @view S[i, j, :]); isnothing(k) ? 1.0 : zᶠ[k]) for i in 1:Nx, j in 1:Ny]
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

groups = Dict("ocean_surface_fields" => "surface",
              "ocean_complete_fields_10year_average_calibrationsample" => "fields",
              "sea_ice_surface_fields" => "seaice",
              "sea_ice_complete_fields_10year_average" => "seaice_mean")
member = Run(dir; grid, groups)
display(member)

fig = dashboard(member; fields = ["surface/T", "fields/T", "fields/S", "seaice/ℵ"],
                section = :x, reference_date = DateTime(1992, 1, 1))

wait(display(fig))
