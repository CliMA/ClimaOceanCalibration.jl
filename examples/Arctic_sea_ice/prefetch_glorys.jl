# Download and inpaint the GLORYS fields panarctic.jl reads into its cache, with the same dates and inpainting.
# The dates can be split over several jobs, job r taking those with index ≡ r (mod njobs):
#   julia +1.13 --project prefetch_glorys.jl --days 365 --njobs 8 --job 0   (… --job 7)

using ArgParse
using NumericalEarth, Oceananigans, Dates, CUDA
using NumericalEarth.DataWrangling: NearestNeighborInpainting
using CopernicusMarine   # enables the GLORYS download extension

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--start_date"
            help = "start date of the run"
            default = "1993-01-02"
        "--days"
            help = "run length in days"
            arg_type = Int
            default = 365
        "--njobs"
            help = "number of jobs the dates are split over"
            arg_type = Int
            default = 1
        "--job"
            help = "index of this job, 0 to njobs - 1"
            arg_type = Int
            default = 0
        "--glorys_dir"
            help = "GLORYS download cache"
            default = "~/GLORYS_data"
        "--architecture"
            help = "GPU or CPU (the inpainting runs here)"
            default = "GPU"
    end
    return parse_args(s)
end

args = parse_commandline()
arch       = args["architecture"] == "GPU" ? GPU() : CPU()
start_date = DateTime(args["start_date"])
dates      = start_date : Day(1) : start_date + Day(args["days"])
mine       = [d for (k, d) in enumerate(dates) if (k - 1) % args["njobs"] == args["job"]]
dataset    = GLORYSDaily()
region     = BoundingBox(longitude = (-180, 180), latitude = (43, 90))
glorys_dir = expanduser(args["glorys_dir"])

fields = ((:temperature, nothing), (:salinity, nothing),
          (:u_velocity, NearestNeighborInpainting(100)), (:v_velocity, NearestNeighborInpainting(100)),
          (:free_surface, NearestNeighborInpainting(100)))

for (n, date) in enumerate(mine)
    t₀ = time()
    for (name, inpainting) in fields
        metadatum = Metadatum(name; date, dataset, region, dir = glorys_dir)
        isnothing(inpainting) ? Field(metadatum, arch) : Field(metadatum, arch; inpainting)
    end
    for name in (:sea_ice_thickness, :sea_ice_concentration)
        Field(Metadatum(name; date, dataset, region, dir = glorys_dir), CPU())
    end
    @info "$(Date(date)) done ($n of $(length(mine)), $(round(Int, time() - t₀)) s)"
    GC.gc()
    arch isa GPU && CUDA.reclaim()
end
