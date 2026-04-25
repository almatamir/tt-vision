#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_analysis.sh  —  Full table-tennis analysis pipeline
#
# Usage:
#   ./run_analysis.sh <video> <start_sec> <end_sec>
#
# Example:
#   ./run_analysis.sh input.mp4 1150 1190
#
# Outputs (all written to current directory):
#   player_position_heatmap.png   — static combined heatmap
#   player_movement_plot.png      — X/Y trajectory over time
#   player_positions.csv          — per-frame player positions
#   output_with_detections.mp4    — bounding-box overlay video
#   heatmap_video.mp4             — progressive heatmap video
#   output_shot/shot_data.csv     — per-shot classification log
#   output_shot/shot_distribution.png
#   output_shot/shot_sequence.png
#   score_video.mp4               — score overlay + title cards + shot labels
#   score_chart.png               — score progression chart
#   score_chart.csv               — score event log (accumulates across runs)
#   score_screenshots/            — OCR proof screenshot per score event
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="/opt/anaconda3/envs/myenv/bin/python"
VIDEO="${1:-input.mp4}"
START_SEC="${2:-1150}"
END_SEC="${3:-1190}"

echo "Video : $VIDEO"
echo "Range : ${START_SEC}s – ${END_SEC}s"
echo ""

# ── Step 1: Player tracking + heatmap ────────────────────────────────────────
echo "━━━  Step 1/3 — Player tracking & heatmap  ━━━"
"$PYTHON" table_tennis_analyzer.py "$VIDEO" \
    --start "$START_SEC" \
    --end   "$END_SEC" \
    --mode  video
echo ""

# ── Step 2: Shot classification ───────────────────────────────────────────────
echo "━━━  Step 2/3 — Shot classification  ━━━"
"$PYTHON" shot_type.py "$VIDEO" \
    --start      "$START_SEC" \
    --end        "$END_SEC" \
    --output-dir output_shot
echo ""

# ── Step 3: Score analysis + video ────────────────────────────────────────────
echo "━━━  Step 3/3 — Score OCR & annotated video  ━━━"
"$PYTHON" score_analyzer.py "$VIDEO" \
    --start      "$START_SEC" \
    --end        "$END_SEC" \
    --shot-data  output_shot/shot_data.csv
echo ""

echo "✓ Done — all outputs written to $(pwd)"
