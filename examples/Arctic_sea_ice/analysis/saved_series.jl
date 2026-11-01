# Daily fields saved by panarctic.jl, read as plain arrays (halo width 7) with their times.
const JLD2 = Oceananigans.OutputWriters.JLD2
const H = 7

iterations(f, name) = sort(parse.(Int, filter(k -> all(isdigit, k), collect(keys(f["timeseries/$name"])))))

function series(file, name; float = Float64)
    JLD2.jldopen(file) do f
        its = iterations(f, name)
        [f["timeseries/t/$it"] for it in its], [float.(Array(f["timeseries/$name/$it"])[H+1:end-H, H+1:end-H, 1]) for it in its]
    end
end
