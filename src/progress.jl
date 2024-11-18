using Statistics
using Oceananigans: prettytime

struct Progress{W}
    wall_clock :: W
end

Progress() = Progress(Ref(time_ns()))

function (progress::Progress)(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    Tmax = maximum(T)
    Tmin = minimum(T)
    umax = maximum(abs, u), maximum(abs, v), maximum(abs, w)

    step_time = 1e-9 * (time_ns() - progress.wall_clock[])

    @info @sprintf("Time: %s, n: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, extrema(T): (%.2f, %.2f) ᵒC, wall time: %s \n",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   umax..., Tmax, Tmin, prettytime(step_time))

     progress.wall_clock[] = time_ns()

     return nothing
end


