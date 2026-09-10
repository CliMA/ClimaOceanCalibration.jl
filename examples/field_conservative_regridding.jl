using Oceananigans
using Oceananigans.Architectures: on_architecture
using ConservativeRegridding
using CUDA

z = (-1, 0)

arch = GPU()
tg = TripolarGrid(arch; size=(360, 170, 1), z, southernmost_latitude = -80)

llg = LatitudeLongitudeGrid(arch; size=(360, 180, 1), z,
                            longitude=(0, 360), latitude=(-82, 90))

src_field = Field{Center, Center, Nothing}(tg)
dst_field = Field{Center, Center, Nothing}(llg)

λ₀, φ₀ = 150, 30.  # degrees
width = 12         # degrees
set!(src_field, (λ, φ) -> exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2))

# The weights are computed on the host, then moved to `arch` alongside the fields.
regridder = ConservativeRegridding.Regridder(on_architecture(CPU(), llg),
                                             on_architecture(CPU(), tg))

regridder = on_architecture(arch, regridder)

ConservativeRegridding.regrid!(dst_field, regridder, src_field)
