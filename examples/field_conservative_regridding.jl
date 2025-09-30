using Oceananigans
using XESMF
using CUDA

z = (-1, 0)

arch = GPU()
tg = TripolarGrid(arch; size=(360, 170, 2), z, southernmost_latitude = -80)

llg = LatitudeLongitudeGrid(arch; size=(360, 180, 2), z,
                            longitude=(0, 360), latitude=(-82, 90))

src_field = CenterField(tg)
dst_field = CenterField(llg)

λ₀, φ₀ = 150, 30.  # degrees
width = 12         # degrees
set!(src_field, (λ, φ, z) -> exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / 2width^2))

regridder = XESMF.Regridder(dst_field, src_field, method="conservative")

regrid!(dst_field, regridder, src_field)