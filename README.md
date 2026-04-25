# AI Table Tennis Analyzer

A multi-module computer vision pipeline for analyzing table-tennis broadcast footage. The system performs real-time player tracking, pose-based shot classification, scoreboard OCR, and progressive heatmap generation — outputting an annotated video with live score overlay and shot labels.

## Architecture

```
Video Input
  ├── table_tennis_analyzer.py  →  Player tracking + position heatmap
  ├── shot_type.py              →  Shot classification (forehand / backhand / smash / push)
  └── score_analyzer.py         →  Scoreboard OCR + annotated output video
```

## Modules

### Player Tracking (`table_tennis_analyzer.py`)
- Detects and tracks players using **YOLOv8 + ByteTrack**
- Validates frames using spatial filters (aspect ratio, separation distance, position jump limits)
- Two-pass pipeline: first pass collects valid frame indices without holding pixel data in memory; second pass renders the heatmap video — peak memory is O(1) regardless of video length
- Builds an incremental **Gaussian density heatmap** via `HeatmapAccumulator` — O(n × W×H) vs the naive O(n² × W×H) of recomputing over all prior positions each frame
- Exports player positions to CSV and generates a static high-resolution heatmap overlay

### Shot Classification (`shot_type.py`)
- Uses **YOLOv8 pose estimation** to extract 17 COCO keypoints per player
- Three-stage detector per shot:
  1. **Confidence gate** — discards frames where YOLO is uncertain about wrist/elbow joints (motion blur, occlusion)
  2. **Peak detection** — fires only when wrist speed has just peaked (previous frame faster than both neighbours), avoiding duplicate triggers across the same swing
  3. **Window vote** — classifies the last N above-threshold frames by majority vote so one noisy frame cannot flip the result
- Classifies shots using arm geometry: wrist position relative to shoulder and body midline determines SMASH / BACKHAND / FOREHAND / PUSH
- Outputs shot distribution chart, time-series scatter plot, and raw CSV

### Score OCR (`score_analyzer.py`)
- Reads the broadcast scoreboard using **EasyOCR** with lazy initialisation (avoids penalising imports that don't need OCR)
- Crops and 2× upscales the relevant screen region before OCR for better accuracy
- Detects score-change events and inserts title cards into the output video
- Saves per-event debug screenshots (full frame + zoomed OCR crops) for auditability
- Exports score progression chart and CSV log

## Setup

```bash
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
bash run_analysis.sh <video_path>
```

**Or run modules individually:**
```bash
# Player tracking and heatmap
python table_tennis_analyzer.py video.mp4 --mode analyze --start 30 --end 90

# Shot classification
python shot_type.py video.mp4 --start 30 --end 90 --output-dir output_shot

# Score OCR and annotated video
python score_analyzer.py video.mp4 --start 30 --end 90
```

## Output Files

| File | Description |
|------|-------------|
| `output_with_detections.mp4` | Annotated video with player bounding boxes |
| `output_with_heatmap.mp4` | Progressive heatmap video |
| `player_position_heatmap.png` | Static high-resolution heatmap |
| `player_positions.csv` | Per-frame player coordinates |
| `output_shot/shot_classification.mp4` | Video with shot labels |
| `output_shot/shot_distribution.png` | Shot type bar chart by player |
| `output_shot/shot_sequence.png` | Shot time-series scatter plot |
| `output_shot/shot_data.csv` | Raw shot event log |
| `score_video.mp4` | Output video with live score overlay and title cards |
| `score_chart.png` | Score progression over time |
| `score_chart.csv` | Score event log |

## Tech Stack

Python · YOLOv8 · ByteTrack · EasyOCR · OpenCV · SciPy · Matplotlib · NumPy · Pandas
