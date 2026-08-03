#!/bin/bash
# download_omip2_raw.sh — plain-bash downloader for the raw OMIP2 flux files.
#
# Reads omip2_download_urls.txt (one line per file: <size> <name> <url> [replicas...],
# generated from omip2_manifest.json by `julia fetch_omip2_fluxes.jl manifest`)
# and fetches everything into RAW_DIR with resume + replica fallback. Needs only
# curl — no Julia or Python. Re-run to resume; complete files are skipped.
#
# Usage:
#   ./download_omip2_raw.sh                 # into /tmp/omip2_raw
#   RAW_DIR=/scratch/omip2_raw ./download_omip2_raw.sh
#
# Afterwards process with:
#   julia --project=<repo> fetch_omip2_fluxes.jl process
#
# Note (July 2026): the CMCC-CM2-SR5, CNRM-CM6-1 and TaiESM1-TIMCOM data nodes
# are offline with no replicas — those lines will fail until the nodes return.
# `-k` skips certificate verification (several ESGF nodes ship broken certs;
# the data is public), matching the stock ESGF wget scripts.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${RAW_DIR:-/tmp/omip2_raw}"
URL_LIST="${URL_LIST:-$HERE/omip2_download_urls.txt}"

[[ -f "$URL_LIST" ]] || { echo "URL list not found: $URL_LIST" >&2; exit 1; }
mkdir -p "$RAW_DIR"

nok=0 nfail=0
while read -r size name urls; do
    [[ "$size" == \#* || -z "$size" ]] && continue
    dest="$RAW_DIR/$name"
    if [[ -f "$dest" ]] && { [[ "$size" == 0 ]] || [[ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")" == "$size" ]]; }; then
        echo "  ok       $name"
        ((nok++)); continue
    fi
    got=false
    for url in $urls; do
        host="${url#*://}"; host="${host%%/*}"
        printf "  fetching %s  (%d MB) from %s\n" "$name" "$((size / 1000000))" "$host"
        if curl -kL -sS --fail --retry 3 -C - -o "$dest" --max-time 7200 "$url" \
           && { [[ "$size" == 0 ]] || [[ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")" == "$size" ]]; }; then
            got=true; break
        fi
        echo "    failed, trying next replica"
    done
    if $got; then ((nok++)); else ((nfail++)); echo "  FAILED   $name"; fi
done < "$URL_LIST"

echo "Download done: $nok ok, $nfail failed  → $RAW_DIR"
exit 0
