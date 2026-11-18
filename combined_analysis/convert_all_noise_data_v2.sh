#!/bin/bash
# Convert every noise_analysis* folder of one recording to noise_data_v2.nc, one file at a time.
# Usage: convert_all_noise_data_v2.sh <zebrafish_DD_MM_YYYY> [pixel size µm]
# Recordings from 20/04/2026 use 2.0 µm stimulus pixels; earlier ones keep their own value.
set -e
cd "$(dirname "$0")/.."
REC="$1"; PX="${2:-}"
for f in /mnt/mea_nas/Laura/"$REC"/Phase_*/noise_analysis*/; do
  [ -f "$f/noise_data.nc" ] || continue
  [ -f "$f/noise_data_v2.nc" ] && { echo "SKIP (exists): ${f#/mnt/mea_nas/Laura/}"; continue; }
  args=("$f"); [ -n "$PX" ] && args+=(--pixel-size "$PX")
  /usr/bin/time -f "  %e s, peak %M KB" poetry run python -m combined_analysis.convert_noise_data_v2 "${args[@]}" 2>&1 \
    | grep -E "Relabelling|Saving|  [0-9.]+ s|Error|Traceback" | sed 's#/mnt/mea_nas/Laura/##'
done
