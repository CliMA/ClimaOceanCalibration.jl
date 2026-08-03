# fetch_omip2_fluxes.jl — download OMIP2 air–sea flux data from ESGF and
# process it into compact monthly-climatology NetCDFs in omip_data/omip2_fluxes/.
# Pure Julia (repo project env); `curl` is the only external tool used.
#
# Three stages (each resumable / skippable):
#
#   julia --project=<repo> fetch_omip2_fluxes.jl manifest   # query ESGF → manifest json
#   julia --project=<repo> fetch_omip2_fluxes.jl download   # fetch raw files (RAW_DIR)
#   julia --project=<repo> fetch_omip2_fluxes.jl process    # → OUT_DIR/*.nc
#   julia --project=<repo> fetch_omip2_fluxes.jl all        # everything
#
# Env vars:
#   RAW_DIR   scratch dir for raw ESGF files (default: /tmp/omip2_raw — NOT the
#             repo; raw files total ~4 GB and can be deleted after processing)
#   OUT_DIR   processed output dir (default: <repo>/omip_data/omip2_fluxes)
#   MANIFEST  manifest path (default: <this dir>/omip2_manifest.json)
#
# What it downloads: the files covering the LAST 31 YEARS of each dataset's
# time axis (models label the 6×61-yr JRA55-do cycles differently: CESM2 uses
# pseudo-years 0001–0366, CMCC 1653–2018, CNRM ships a real-labelled
# final-cycle file 1958–2018). The processing step then averages the last 360
# months into a 12-month climatology — the same "last 360 months of final
# cycle" convention as the existing omip_data/ tos/t200 maps.
#
# What it writes per (variable, model): <var>_omip2_<MODEL>.nc with
#   clim (12, ny, nx) native climatology, lat/lon, and clim_r (12, 180, 360),
#   a 1° bin-average regrid (same operator as postprocess_flux_climatology.jl).
#
# Focus variables (per model availability on ESGF, July 2026):
#   hfds rsntds rlntds evs prra prsn wfo
# (hfls/hfss are NOT published for omip2 by any model; latent heat can be
# derived as Lv*evs where evs exists. The CMCC, CNRM and TaiESM1 data nodes
# are currently offline with no replicas — their specs stay in the manifest
# but downloads fail until the nodes return.)

using JSON
using NCDatasets
using Dates
using Printf

include(joinpath(@__DIR__, "binavg_regrid.jl"))

const HERE = @__DIR__
const REPO = dirname(dirname(HERE))
const RAW_DIR  = get(ENV, "RAW_DIR", "/tmp/omip2_raw")
const OUT_DIR  = get(ENV, "OUT_DIR", joinpath(REPO, "omip_data", "omip2_fluxes"))
const MANIFEST = get(ENV, "MANIFEST", joinpath(HERE, "omip2_manifest.json"))

# Independent ESGF Solr indexes; distributed queries fan out to remote shards
# and time out intermittently, so we alternate between them per retry. The LiU
# (Sweden) index locally holds EC-Earth3 and NorESM2-LM omip2 file records.
const BASES = ["https://esgf-data.dkrz.de/esg-search/search",
               "https://esg-dn1.nsc.liu.se/esg-search/search",
               "https://esgf.ceda.ac.uk/esg-search/search"]

const SPECS = [
    ("CESM2",          "r1i1p1f1", "gr", ["hfds", "rsntds", "rlntds", "evs"]),
    ("CMCC-CM2-SR5",   "r1i1p1f1", "gn", ["hfds", "rsntds", "rlntds", "evs", "prra", "prsn", "wfo"]),
    ("CNRM-CM6-1",     "r1i1p1f2", "gn", ["hfds", "rsntds", "wfo"]),
    ("EC-Earth3",      "r1i1p1f1", "gn", ["hfds", "rsntds", "evs", "prsn", "wfo"]),
    ("NorESM2-LM",     "r1i1p1f1", "gn", ["hfds", "rsntds", "evs", "prra", "prsn", "wfo"]),
    ("TaiESM1-TIMCOM", "r1i1p1f1", "gn", ["hfds", "wfo"]),
]

# _(YYYYMM)-(YYYYMM).nc → (start, stop) as Ints, or nothing
function frange(fname)
    m = match(r"_(\d{6})-(\d{6})\.nc$", fname)
    return m === nothing ? nothing : (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

# ──────────────────────────────────────────────────────────────────
# Stage 1: manifest
# ──────────────────────────────────────────────────────────────────
# Several ESGF nodes have broken/expired certificates; the data is public, so
# `curl -k` skips verification (as the stock ESGF wget scripts do).
function esgf_query(params::Vector{Pair{String, String}}; retries = 10, sleeptime = 3)
    query = join(["$(k)=$(v)" for (k, v) in vcat(params, ["format" => "application/solr+json"])], "&")
    for i in 1:retries
        base = BASES[mod1(i, length(BASES))]
        url = base * "?" * query
        out = IOBuffer(); err = IOBuffer()
        p = run(pipeline(ignorestatus(`curl -ksS --max-time 45 $url`); stdout = out, stderr = err), wait = true)
        if success(p)
            d = try
                JSON.parse(String(take!(out)))
            catch
                nothing
            end
            if d !== nothing && get(get(d, "response", Dict()), "numFound", 0) > 0
                return d
            end
        else
            println("    retry $i ($(split(base, '/')[3])): $(first(String(take!(err)), 60))")
        end
        sleep(sleeptime)
    end
    return nothing
end

function build_manifest()
    manifest = isfile(MANIFEST) ? JSON.parsefile(MANIFEST) : Any[]
    done = Set((m["source_id"], m["variable"]) for m in manifest if !isempty(m["files"]))
    for (source_id, member, grid, variables) in SPECS
        for var in variables
            if (source_id, var) in done
                @printf("%-16s %-7s: already in manifest\n", source_id, var)
                continue
            end
            @printf("%-16s %-7s: querying...\n", source_id, var)
            base_params = ["project" => "CMIP6", "experiment_id" => "omip2", "table_id" => "Omon",
                           "source_id" => source_id, "variant_label" => member, "grid_label" => grid,
                           "variable_id" => var, "type" => "File", "limit" => "800",
                           "fields" => "title,url,size,checksum,checksum_type"]
            d = nothing
            for (distrib, retries) in (("false", 4), ("true", 20))  # local index first
                d = esgf_query(vcat(base_params, ["distrib" => distrib]); retries)
                d === nothing || break
            end
            if d === nothing
                @printf("%-16s %-7s: NOT FOUND\n", source_id, var)
                continue
            end
            seen = Dict{String, Dict{String, Any}}()
            for doc in d["response"]["docs"]
                title = doc["title"]
                http = [split(u, "|")[1] for u in get(doc, "url", []) if endswith(u, "HTTPServer")]
                isempty(http) && continue
                e = get!(seen, title,
                         Dict("file" => title, "size" => get(doc, "size", 0), "urls" => String[]))
                for u in http
                    u in e["urls"] || push!(e["urls"], u)
                end
            end
            files = sort(collect(values(seen)); by = f -> f["file"])
            ends = [frange(f["file"])[2] for f in files if frange(f["file"]) !== nothing]
            isempty(ends) && continue
            last_end = maximum(ends)
            cutoff = (last_end ÷ 100 - 31) * 100 + last_end % 100
            keep = [f for f in files if frange(f["file"]) !== nothing && frange(f["file"])[2] > cutoff]
            tot = sum(f["size"] for f in keep) / 1e6
            @printf("%-16s %-7s: keep %d/%d files, %.0f MB\n", source_id, var, length(keep), length(files), tot)
            manifest = [m for m in manifest if !(m["source_id"] == source_id && m["variable"] == var)]
            push!(manifest, Dict("source_id" => source_id, "variable" => var, "grid" => grid,
                                 "member" => member, "files" => keep))
            open(MANIFEST, "w") do io
                JSON.print(io, manifest, 1)
            end
        end
    end
    nresolved = count(m -> !isempty(m["files"]), manifest)
    total = sum(Float64(f["size"]) for m in manifest for f in m["files"]; init = 0.0) / 1e9
    @printf("Manifest: %d specs, %.1f GB\n", nresolved, total)
end

# ──────────────────────────────────────────────────────────────────
# Stage 2: download
# ──────────────────────────────────────────────────────────────────
function download_raw()
    manifest = JSON.parsefile(MANIFEST)
    mkpath(RAW_DIR)
    nok = nfail = 0
    for m in manifest
        for f in m["files"]
            dest = joinpath(RAW_DIR, f["file"])
            want = f["size"]
            if isfile(dest) && (want == 0 || filesize(dest) == want)
                println("  ok       $(f["file"])")
                nok += 1
                continue
            end
            got = false
            for url in f["urls"]
                @printf("  fetching %s  (%.0f MB) from %s\n", f["file"], want / 1e6, split(url, '/')[3])
                p = run(ignorestatus(`curl -kL -sS --fail --retry 3 -C - -o $dest --max-time 7200 $url`))
                if success(p) && (want == 0 || filesize(dest) == want)
                    got = true
                    break
                end
                println("    failed (rc=$(p.exitcode)), trying next replica")
            end
            got ? (nok += 1) : begin
                nfail += 1
                println("  FAILED   $(f["file"])")
            end
        end
    end
    println("Download done: $nok ok, $nfail failed")
    return nfail
end

# ──────────────────────────────────────────────────────────────────
# Stage 3: process → climatology + 1° regrid
# ──────────────────────────────────────────────────────────────────

# Find the variable's lat/lon coordinate arrays by standard_name/units,
# returned as (nx, ny) matrices (NCDatasets reads C-ordered files reversed).
function native_coords(ds, var)
    coord_names = split(get(ds[var].attrib, "coordinates", ""))
    latn = lonn = nothing
    for name in vcat(String.(coord_names), keys(ds))
        haskey(ds, name) || continue
        sn = get(ds[name].attrib, "standard_name", "")
        un = get(ds[name].attrib, "units", "")
        latn === nothing && (sn == "latitude"  || un == "degrees_north") && (latn = name)
        lonn === nothing && (sn == "longitude" || un == "degrees_east")  && (lonn = name)
    end
    (latn === nothing || lonn === nothing) && error("could not find lat/lon coordinates for $var")
    lat = Float64.(coalesce.(Array(ds[latn]), NaN))
    lon = Float64.(coalesce.(Array(ds[lonn]), NaN))
    if ndims(lat) == 1   # regular grid (e.g. CESM2 gr) → broadcast to (nx, ny)
        lon2 = [λ for λ in lon, _ in lat]
        lat2 = [φ for _ in lon, φ in lat]
        return lat2, lon2
    end
    return lat, lon
end

function process()
    mkpath(OUT_DIR)
    manifest = JSON.parsefile(MANIFEST)

    for m in manifest
        var, model = m["variable"], m["source_id"]
        out_path = joinpath(OUT_DIR, "$(var)_omip2_$(model).nc")
        if isfile(out_path)
            @printf("%-16s %-7s: processed file exists, skipping\n", model, var)
            continue
        end
        paths = [joinpath(RAW_DIR, f["file"]) for f in sort(m["files"]; by = f -> f["file"])]
        nmissing = count(!isfile, paths)
        if nmissing > 0
            @printf("%-16s %-7s: %d raw files missing, skipping\n", model, var, nmissing)
            continue
        end

        # Concatenate the (already last-31-yr) files along time; keep last 360 months
        chunks = Array{Float64, 3}[]   # each (nt, nx, ny)
        months = Int[]
        lat = lon = nothing
        for p in paths
            NCDataset(p) do ds
                A = Float64.(coalesce.(Array(ds[var]), NaN))      # (nx, ny, nt)
                push!(chunks, permutedims(A, (3, 1, 2)))          # → (nt, nx, ny)
                append!(months, Dates.month.(Array(ds["time"])))
                if lat === nothing
                    lat, lon = native_coords(ds, var)
                end
            end
        end
        data = vcat(chunks...)
        nt = size(data, 1)
        nt < 360 && @printf("%-16s %-7s: only %d months available (<360), using all\n", model, var, nt)
        nuse = min(nt, 360)
        data   = data[end-nuse+1:end, :, :]
        months = months[end-nuse+1:end]

        nx, ny = size(data, 2), size(data, 3)
        clim = fill(NaN, 12, nx, ny)
        for mm in 1:12
            sel = findall(==(mm), months)
            isempty(sel) && continue
            for j in 1:ny, i in 1:nx
                s = 0.0; n = 0
                for t in sel
                    v = data[t, i, j]
                    isnan(v) && continue
                    s += v; n += 1
                end
                n > 0 && (clim[mm, i, j] = s / n)
            end
        end

        wet = [isfinite(clim[1, i, j]) for i in 1:nx, j in 1:ny]
        clim_r = regrid_binavg_1deg(clim, lon, lat, wet)

        write_climatology_netcdf(out_path, var, model, m["member"], clim, lon, lat, clim_r;
                                 clim_months = nuse,
                                 note = "monthly climatology of last $nuse months of final JRA55-do cycle; " *
                                        "processed by fetch_omip2_fluxes.jl")
        @printf("%-16s %-7s: -> %s\n", model, var, basename(out_path))
    end
end

# ──────────────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    stage = isempty(ARGS) ? "all" : ARGS[1]
    stage in ("manifest", "download", "process", "all") ||
        error("usage: fetch_omip2_fluxes.jl [manifest|download|process|all]")
    stage in ("manifest", "all") && build_manifest()
    stage in ("download", "all") && download_raw()
    stage in ("process", "all")  && process()
end
