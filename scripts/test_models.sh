#!/usr/bin/env bash
# Run scoring on up to 5 test videos for each exercise (squat, pushup, lunge)
# Usage: ./scripts/test_models.sh
#
# Windows / PowerShell notes:
#  - This script is a Unix-style bash script. If you're on Windows, you can run it
#    using Git Bash / WSL / MSYS environments that provide `bash`.
#  - If you prefer PowerShell (no bash), here's an equivalent snippet you can run
#    from PowerShell (run from the repository root). It finds up to 5 videos per
#    exercise and calls the Python scorer:
#
#    $root = (Get-Location).Path
#    foreach ($exercise in 'squat','pushup','lunge') {
#      switch ($exercise) {
#        'squat'  { $dir = Join-Path $root 'data/squats_test' }
#        'pushup' { $dir = Join-Path $root 'data/PushUps_test' }
#        'lunge'  { $dir = Join-Path $root 'data/lunges_test' }
#      }
#      if (-Not (Test-Path $dir)) { Write-Host "Skipping - test dir not found: $dir"; continue }
#      Get-ChildItem -Path $dir -File -Include *.mp4,*.mov,*.avi,*.mkv | Select-Object -First 5 |
#        ForEach-Object { Write-Host "-> Scoring $($_.FullName)"; python "$root\scripts\score_video.py" $_.FullName --exercise $exercise }
#    }
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="python"

declare -A TEST_DIRS
TEST_DIRS[squat]="${ROOT_DIR}/data/squats_test"
TEST_DIRS[pushup]="${ROOT_DIR}/data/PushUps_test"
TEST_DIRS[lunge]="${ROOT_DIR}/data/lunges_test"

echo "Running up to 5 test videos per model using scripts/score_video.py"

for exercise in "squat" "pushup" "lunge"; do
  dir=${TEST_DIRS[$exercise]}
  echo
  echo "=== Exercise: $exercise (dir: $dir) ==="
  if [ ! -d "$dir" ]; then
    echo "Skipping — test dir not found: $dir"
    continue
  fi

  # find up to 5 video files
  IFS=$'\n' read -r -d '' -a vids < <(find "$dir" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \) | head -n 5 && printf '\0')

  if [ ${#vids[@]} -eq 0 ]; then
    echo "No video files found in $dir"
    continue
  fi

  for v in "${vids[@]}"; do
    echo "-> Scoring $v"
    # run the Python scorer with explicit exercise argument
    "$PY" "$ROOT_DIR/scripts/score_video.py" "$v" --exercise "$exercise" || echo "scoring failed for $v"
  done
done

echo
echo "All done."
