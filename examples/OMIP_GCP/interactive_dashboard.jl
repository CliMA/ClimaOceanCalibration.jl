# Interactive dashboard for one or more OMIP run directories:
#
#     julia --project examples/OMIP_GCP/interactive_dashboard.jl path/to/a_run [path/to/b_run]
#
# From a REPL, `Run(dir)` lists the available `group/variable` keys, and
# `dashboard(runs...; fields = [...])` opens the figure.

using GLMakie
using Dates
using ClimaOceanCalibration.Visualization

runs = [Run(dir) for dir in ARGS]
display(first(runs))

fig = dashboard(runs...; fields = ["surface/tos", "fields/to", "fields/so"],
                section = :x, reference_date = DateTime(1958, 1, 1))

wait(display(fig))
