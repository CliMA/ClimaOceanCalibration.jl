
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Statistics
using OrderedCollections

filename = "north_atlantic_calibration_i0_e0.jld2"

Tt = FieldTimeSeries(filename, "T")
St = FieldTimeSeries(filename, "S")

fig = Figure()
axT = Axis(fig[1, 1])
axS = Axis(fig[1, 2])

Nz = size(T, 3)
heatmap!(axT, view(Tt[2], :, :, Nz))
heatmap!(axS, view(St[2], :, :, Nz))

display(fig)

