# The member with the default coefficients, whose residual `estimate_redundancy.jl` reads.
#
#   julia --project examples/CATKE_GM_decadal_calibration/run_default_member.jl <member directory>

include("forward_model_onedegree.jl")

run_decadal_member(Dict{String, Float64}(); directory = ARGS[1])
