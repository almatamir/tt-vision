from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


class PersonDetection(NamedTuple):
    track_id: int
    cx:       float
    cy:       float
    w:        float
    h:        float


class PlayerPosition(NamedTuple):
    x: int
    y: int


class TrajectoryPoint(NamedTuple):
    frame: int
    x:     int
    y:     int


@dataclass
class AnalysisResult:
    p1_positions:     list[PlayerPosition]
    p2_positions:     list[PlayerPosition]
    p1_trajectory:    list[TrajectoryPoint]
    p2_trajectory:    list[TrajectoryPoint]
    background_frame: np.ndarray | None
    width:            int
    height:           int


@dataclass
class AnalyzerConfig:
    # Detection filters (pixel values assume ~1080p source)
    min_player_width:    int   = 20
    min_player_height:   int   = 60
    y_min:               int   = 100
    y_max:               int   = 700

    # Frame validity checks
    min_separation:      int   = 150
    min_aspect_ratio:    float = 1.2
    max_position_jump:   int   = 150
    y_mean_min:          int   = 200
    y_mean_max:          int   = 650

    # Heatmap rendering
    heatmap_sigma:       float = 20.0
    heatmap_sigma_static: float = 30.0
    heatmap_threshold:   float = 0.1
