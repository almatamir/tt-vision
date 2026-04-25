"""
Shared data types and configuration for the table-tennis analysis pipeline.

Centralising types here prevents circular imports between modules and ensures
every threshold lives in one place — change it once, it takes effect everywhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np


# ── Immutable value types ──────────────────────────────────────────────────────

class PersonDetection(NamedTuple):
    """A single person bounding-box detected in one video frame."""
    track_id: int
    cx:       float   # centre-x  (pixels)
    cy:       float   # centre-y  (pixels)
    w:        float   # box width
    h:        float   # box height


class PlayerPosition(NamedTuple):
    """Pixel-space position used for heatmap accumulation."""
    x: int
    y: int


class TrajectoryPoint(NamedTuple):
    """One sample of a player's on-court position over time."""
    frame: int
    x:     int
    y:     int


# ── Analysis result ────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Everything produced by a full analyze_video() run.

    Using a dataclass instead of a bare 6-tuple means callers refer to fields
    by name — result.p1_positions rather than result[0] — so the contract is
    explicit and resistant to accidental index misordering.
    """
    p1_positions:    list[PlayerPosition]
    p2_positions:    list[PlayerPosition]
    p1_trajectory:   list[TrajectoryPoint]
    p2_trajectory:   list[TrajectoryPoint]
    background_frame: np.ndarray | None
    width:           int
    height:          int


# ── Centralised configuration ──────────────────────────────────────────────────

@dataclass
class AnalyzerConfig:
    """
    All detection and validation thresholds in one place.

    Pixel values assume a ~1080p source.  For other resolutions pass a custom
    instance with values scaled proportionally.
    """
    # Person-detection filters
    min_player_width:   int   = 20
    min_player_height:  int   = 60
    y_min:              int   = 100    # ignore detections above this row
    y_max:              int   = 700    # ignore detections below this row

    # Frame-validity checks
    min_separation:     int   = 150    # minimum horizontal gap between players (px)
    min_aspect_ratio:   float = 1.2    # minimum height/width for a player box
    max_position_jump:  int   = 150    # maximum displacement between valid frames (px)
    y_mean_min:         int   = 200    # valid range for the mean player Y position
    y_mean_max:         int   = 650

    # Heatmap rendering
    heatmap_sigma:      float = 20.0   # gaussian blur radius (progressive video)
    heatmap_sigma_static: float = 30.0 # gaussian blur radius (static image)
    heatmap_threshold:  float = 0.1    # density below this is rendered transparent
