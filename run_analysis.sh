#!/usr/bin/env bash
set -euo pipefail

PYTHON="/opt/anaconda3/envs/myenv/bin/python"
VIDEO="${1:-input.mp4}"
START_SEC="${2:-1150}"
END_SEC="${3:-1190}"

echo "Video: $VIDEO  |  Range: ${START_SEC}s - ${END_SEC}s"

echo "Step 1/3: Player tracking and heatmap..."
"$PYTHON" table_tennis_analyzer.py "$VIDEO" \
    --start "$START_SEC" \
    --end   "$END_SEC" \
    --mode  video

echo "Step 2/3: Shot classification..."
"$PYTHON" shot_type.py "$VIDEO" \
    --start      "$START_SEC" \
    --end        "$END_SEC" \
    --output-dir output_shot

echo "Step 3/3: Score OCR and annotated video..."
"$PYTHON" score_analyzer.py "$VIDEO" \
    --start     "$START_SEC" \
    --end       "$END_SEC" \
    --shot-data output_shot/shot_data.csv

echo "Done. All outputs written to $(pwd)"
