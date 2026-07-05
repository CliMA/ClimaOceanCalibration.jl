# binavg_regrid.jl — shared 1° bin-average regridding used by
# postprocess_flux_climatology.jl (our ORCA runs) and fetch_omip2_fluxes.jl
# (OMIP2 native grids). Using the same operator for both sides keeps the
# comparison methodologically uniform.

const LAT_R = collect(-89.5:1.0:89.5)
const LON_R = collect(0.5:1.0:359.5)

"""
    regrid_binavg_1deg(clim, lon, lat, wet) -> Array{Float64,3} (12, 180, 360)

Bin-average a monthly climatology `clim :: (12, nx, ny)` on a (possibly
curvilinear) source grid with cell-center coordinates `lon`, `lat :: (nx, ny)`
and wetness mask `wet :: (nx, ny)` onto the 1° lat-lon grid
(`LAT_R` × `LON_R`). Bins no source cell lands in (≈1° source grids do not
tile the target boxes exactly) are filled from their 8-neighbourhood
(lon-periodic) where ≥3 neighbours are valid, in two passes — closing 1–2-bin
gaps without spreading far into land.
"""
function regrid_binavg_1deg(clim, lon, lat, wet)
    nlat, nlon = length(LAT_R), length(LON_R)
    out = fill(NaN, 12, nlat, nlon)
    cnt = zeros(Int, nlat, nlon)
    acc = zeros(Float64, 12, nlat, nlon)
    nx, ny = size(lon)
    for j in 1:ny, i in 1:nx
        wet[i, j] || continue
        isfinite(clim[1, i, j]) || continue
        (isfinite(lon[i, j]) && isfinite(lat[i, j])) || continue
        λ = mod(lon[i, j], 360.0)
        φ = lat[i, j]
        bi = clamp(floor(Int, λ) + 1, 1, nlon)
        bj = clamp(floor(Int, φ + 90) + 1, 1, nlat)
        cnt[bj, bi] += 1
        for mm in 1:12
            acc[mm, bj, bi] += clim[mm, i, j]
        end
    end
    for bi in 1:nlon, bj in 1:nlat
        cnt[bj, bi] == 0 && continue
        for mm in 1:12
            out[mm, bj, bi] = acc[mm, bj, bi] / cnt[bj, bi]
        end
    end

    for _ in 1:2
        filled = copy(out)
        for bi in 1:nlon, bj in 1:nlat
            isnan(out[1, bj, bi]) || continue
            for mm in 1:12
                s = 0.0; n = 0
                for dj in -1:1, di in -1:1
                    (di == 0 && dj == 0) && continue
                    bj2 = bj + dj
                    1 ≤ bj2 ≤ nlat || continue
                    bi2 = mod1(bi + di, nlon)
                    v = out[mm, bj2, bi2]
                    isnan(v) && continue
                    s += v; n += 1
                end
                n ≥ 3 && (filled[mm, bj, bi] = s / n)
            end
        end
        out = filled
    end
    return out
end

"""
    write_climatology_netcdf(path, var, label, member, clim, lon, lat, clim_r; attrs...)

Write the processed-climatology NetCDF layout shared by the OMIP2 fetch and
our-run postprocess:
    clim   (month, y, x) native monthly climatology   [C order]
    lat/lon (y, x) native coordinates
    clim_r (month, lat_r, lon_r) 1° regrid, lat_r/lon_r axes.
`clim` is passed in Julia layout (12, nx, ny), `clim_r` as (12, 180, 360);
NCDatasets lists dimensions in Julia (Fortran) order — reversed on disk — so
arrays are permuted here to make the on-disk layout identical to files written
by C-order writers.
"""
function write_climatology_netcdf(path, var, label, member, clim, lon, lat, clim_r;
                                  clim_months, note, extra_attrs = Pair{String, Any}[])
    c_xym  = permutedims(clim, (2, 3, 1))     # (12, nx, ny) → (nx, ny, 12)
    cr_xym = permutedims(clim_r, (3, 2, 1))   # (12, 180, 360) → (360, 180, 12)
    NCDataset(path, "c") do ds
        defDim(ds, "month", 12); defDim(ds, "y", size(c_xym, 2)); defDim(ds, "x", size(c_xym, 1))
        defDim(ds, "lat_r", length(LAT_R)); defDim(ds, "lon_r", length(LON_R))
        defVar(ds, "clim",   Float32.(c_xym), ("x", "y", "month"))
        defVar(ds, "lat",    Float32.(lat), ("x", "y"))
        defVar(ds, "lon",    Float32.(lon), ("x", "y"))
        defVar(ds, "clim_r", Float32.(cr_xym), ("lon_r", "lat_r", "month"))
        defVar(ds, "lat_r",  Float32.(LAT_R), ("lat_r",))
        defVar(ds, "lon_r",  Float32.(LON_R), ("lon_r",))
        ds.attrib["source_id"]   = label
        ds.attrib["experiment"]  = "omip2"
        ds.attrib["variable_id"] = var
        ds.attrib["member"]      = member
        ds.attrib["clim_months"] = clim_months
        ds.attrib["note"]        = note
        for (k, v) in extra_attrs
            ds.attrib[k] = v
        end
    end
    return path
end
