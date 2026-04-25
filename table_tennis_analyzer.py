"""
Table-tennis player tracking, position analysis, and heatmap generation.
"""
from __future__ import annotations

import argparse
import csv
import logging
from contextlib import contextmanager
from collections import deque
from typing import Generator, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

from models import (
    AnalysisResult, AnalyzerConfig,
    PersonDetection, PlayerPosition, TrajectoryPoint,
)

logger = logging.getLogger(__name__)

# ── Module-level constants ─────────────────────────────────────────────────────

START_FRAME  = 8000
END_FRAME    = 10000
MODEL_PATH   = "yolov8n-pose.pt"

# Built once at import time — creating a colormap is not free.
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'tt_heatmap',
    [(0, 0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    N=256,
)
_STATIC_CMAP = LinearSegmentedColormap.from_list(
    'tt_static',
    [(1, 1, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    N=256,
)


# ── Resource-management helpers ────────────────────────────────────────────────

@contextmanager
def open_video(path: str) -> Generator[cv2.VideoCapture, None, None]:
    """Guarantee VideoCapture.release() even if the body raises."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        yield cap
    finally:
        cap.release()


@contextmanager
def open_writer(path: str, fourcc: str, fps: float,
                size: tuple[int, int]) -> Generator[cv2.VideoWriter, None, None]:
    """Guarantee VideoWriter.release() even if the body raises."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    try:
        if not out.isOpened():
            raise IOError(f"Cannot open VideoWriter: {path}")
        yield out
    finally:
        out.release()


# ── Incremental heatmap ────────────────────────────────────────────────────────

class HeatmapAccumulator:
    """
    Incrementally builds a two-player position-density heatmap.

    Design rationale
    ----------------
    A naive progressive heatmap recomputes gaussian_filter over *all* prior
    positions on every frame — O(n²) total work.  Instead this class keeps two
    raw accumulator arrays.  Adding a position is O(1) (one array increment).
    Rendering applies a single gaussian_filter pass — O(W×H) — regardless of
    how many positions have been accumulated.  Total cost is O(n × W×H), a
    factor-of-n improvement for any non-trivial video.
    """

    def __init__(self, width: int, height: int,
                 sigma: float = 20.0,
                 threshold: float = 0.1) -> None:
        self.width     = width
        self.height    = height
        self.sigma     = sigma
        self.threshold = threshold
        self._p1 = np.zeros((height, width), dtype=np.float32)
        self._p2 = np.zeros((height, width), dtype=np.float32)

    def add(self, p1: PlayerPosition, p2: PlayerPosition) -> None:
        """Accumulate one frame's positions — O(1)."""
        for pos, arr in ((p1, self._p1), (p2, self._p2)):
            if 0 <= pos.x < self.width and 0 <= pos.y < self.height:
                arr[pos.y, pos.x] += 1.0

    def render(self, cmap=None) -> np.ndarray:
        """
        Smooth, normalise, and return a BGR overlay image — O(W×H).
        Low-density pixels are zeroed in one vectorised operation.
        """
        cmap = cmap or _HEATMAP_CMAP
        blurred  = gaussian_filter(self._p1, self.sigma) + \
                   gaussian_filter(self._p2, self.sigma)
        max_val  = blurred.max()
        if max_val == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        norm     = blurred / max_val * 10
        img      = (cmap(norm / 10)[:, :, :3] * 255).astype(np.uint8)
        img      = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img[norm <= self.threshold] = 0          # vectorised — no pixel loop
        return img

    def combined_density(self, sigma: Optional[float] = None) -> np.ndarray:
        """Return the normalised combined density array for matplotlib overlays."""
        s        = sigma or self.sigma
        combined = gaussian_filter(self._p1, s) + gaussian_filter(self._p2, s)
        max_val  = combined.max()
        return combined / max_val * 10 if max_val > 0 else combined


# ── Detection primitives ───────────────────────────────────────────────────────

def extract_people(boxes,
                   cfg: AnalyzerConfig = AnalyzerConfig()) -> list[PersonDetection]:
    """
    Extract valid person detections from YOLO boxes, sorted left-to-right.
    Handles both tracked (boxes.id present) and untracked results.
    Single authoritative implementation — shot_type.py delegates here
    instead of duplicating this logic.
    """
    people: list[PersonDetection] = []

    if hasattr(boxes, 'id') and boxes.id is not None:
        ids    = boxes.id.cpu().numpy().astype(int)
        bboxes = boxes.xyxy.cpu().numpy()
        cls    = boxes.cls.cpu().numpy()
        src    = zip(ids, bboxes, cls)
    elif len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        cls    = boxes.cls.cpu().numpy()
        src    = ((i, b, c) for i, (b, c) in enumerate(zip(bboxes, cls)))
    else:
        return people

    for track_id, bbox, cl in src:
        if cl != 0:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h   = x2 - x1, y2 - y1
        if (w > cfg.min_player_width and h > cfg.min_player_height
                and cfg.y_min < cy < cfg.y_max):
            people.append(PersonDetection(int(track_id), cx, cy, w, h))

    people.sort(key=lambda p: p.cx)
    return people


def validate_and_extract(
        detections,
        history: Optional[deque] = None,
        cfg: AnalyzerConfig = AnalyzerConfig(),
        check_jump: bool = True,
) -> Optional[tuple[PersonDetection, PersonDetection]]:
    """
    Validate a frame and return the two player detections in one pass.

    Returns (left_player, right_player) if valid, None otherwise.
    Callers use the returned detections directly — no need to call
    extract_people() a second time after checking validity.
    """
    if detections is None or len(detections[0].boxes) == 0:
        logger.debug("No detections")
        return None

    people = extract_people(detections[0].boxes, cfg)
    logger.debug("Detected %d people", len(people))

    if len(people) < 2:
        return None

    p1, p2 = people[0], people[-1]

    if p2.cx - p1.cx < cfg.min_separation:
        logger.debug("Players too close (%.1f px)", p2.cx - p1.cx)
        return None

    if any(p.h / p.w < cfg.min_aspect_ratio for p in people):
        logger.debug("Invalid aspect ratio")
        return None

    if check_jump and history and len(history) >= 2:
        prev_p1, prev_p2 = history[-1]
        if (np.hypot(p1.cx - prev_p1.x, p1.cy - prev_p1.y) > cfg.max_position_jump or
                np.hypot(p2.cx - prev_p2.x, p2.cy - prev_p2.y) > cfg.max_position_jump):
            logger.debug("Position jump too large")
            return None

    y_mean = np.mean([p.cy for p in people])
    if not (cfg.y_mean_min < y_mean < cfg.y_mean_max):
        logger.debug("Players at invalid height (%.1f)", y_mean)
        return None

    return p1, p2


def is_valid_frame(frame, detections,
                   history: Optional[deque] = None,
                   cfg: AnalyzerConfig = AnalyzerConfig()) -> bool:
    """
    Boolean wrapper around validate_and_extract.
    Kept as a public API so shot_type.py can import it without also
    receiving the detection data it does not need.
    """
    return validate_and_extract(detections, history, cfg) is not None


# ── First pass — frame index collection ───────────────────────────────────────

def collect_valid_frames(
        video_path: str,
        start_frame: int = START_FRAME,
        end_frame: int   = END_FRAME,
        model: Optional[YOLO] = None,
        cfg: AnalyzerConfig   = AnalyzerConfig(),
        check_jump: bool      = True,
) -> tuple[list[PlayerPosition], list[PlayerPosition], list[int], float, int, int]:
    """
    Scan the video once and record metadata for valid frames.

    Returns
    -------
    p1_positions, p2_positions, frame_indices, fps, width, height

    No frame pixel data is kept in RAM — only positions and indices.
    fps is returned here so callers never need to re-open the file just
    to read a single property.
    """
    model = model or YOLO(MODEL_PATH)

    with open_video(video_path) as cap:
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame    = end_frame if end_frame > 0 else total_frames

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        p1_positions:  list[PlayerPosition] = []
        p2_positions:  list[PlayerPosition] = []
        frame_indices: list[int]            = []
        history:       deque                = deque(maxlen=5)
        valid_count = 0
        frame_idx   = start_frame

        logger.info("Scanning %s (frames %d–%d)...", video_path, start_frame, end_frame)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > end_frame:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            players = validate_and_extract(results, history, cfg, check_jump)

            if players:
                valid_count += 1
                p1_det, p2_det = players
                p1 = PlayerPosition(int(p1_det.cx), int(p1_det.cy))
                p2 = PlayerPosition(int(p2_det.cx), int(p2_det.cy))
                p1_positions.append(p1)
                p2_positions.append(p2)
                frame_indices.append(frame_idx)
                history.append((p1, p2))

            frame_idx += 1
            if (frame_idx - start_frame) % 50 == 0:
                logger.info("  %d/%d frames — %d valid",
                            frame_idx - start_frame,
                            end_frame - start_frame,
                            valid_count)

    logger.info("Scan complete: %d valid frames", valid_count)
    return p1_positions, p2_positions, frame_indices, fps, width, height


# ── Second pass — progressive heatmap video ───────────────────────────────────

def render_heatmap_video(
        video_path:    str,
        frame_indices: list[int],
        p1_positions:  list[PlayerPosition],
        p2_positions:  list[PlayerPosition],
        fps:           float,
        width:         int,
        height:        int,
        output:        str = "output_with_heatmap.mp4",
        frame_skip:    int = 1,
) -> None:
    """
    Re-read valid frames from disk and write them with a progressive heatmap.

    Separating collection (first pass) from rendering (second pass) means we
    never hold all decoded frames in RAM simultaneously — peak memory is O(1)
    in the number of frames regardless of video length.

    The HeatmapAccumulator is updated incrementally each frame, so total
    gaussian-filter work is O(n × W×H) rather than O(n² × W×H).
    """
    if not frame_indices:
        logger.warning("No valid frames to render")
        return

    heatmap          = HeatmapAccumulator(width, height, sigma=20)
    pos_by_index     = dict(zip(frame_indices, zip(p1_positions, p2_positions)))
    rendered         = 0

    logger.info("Rendering heatmap video (%d frames → %s)...", len(frame_indices), output)

    with open_video(video_path) as cap, \
         open_writer(output, 'mp4v', fps, (width, height)) as out:

        for i, frame_idx in enumerate(frame_indices):
            if frame_skip > 1 and i % frame_skip != 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Could not read frame %d — skipping", frame_idx)
                continue

            p1, p2 = pos_by_index[frame_idx]
            heatmap.add(p1, p2)
            overlay = heatmap.render()

            gray     = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.merge([mask, mask, mask])
            blended  = np.where(mask_3ch == 255,
                                cv2.addWeighted(frame, 0.7, overlay, 0.3, 0),
                                frame)
            cv2.putText(blended, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(blended)
            rendered += 1

            if rendered % 50 == 0:
                logger.info("  Rendered %d/%d", rendered, len(frame_indices))

    logger.info("Heatmap video saved to %s", output)


# ── Full analysis pipeline ─────────────────────────────────────────────────────

def analyze_video(
        video_path:  str,
        start_frame: int  = START_FRAME,
        end_frame:   int  = END_FRAME,
        model: Optional[YOLO]         = None,
        cfg:   AnalyzerConfig         = AnalyzerConfig(),
) -> AnalysisResult:
    """
    Run detection on every frame, write annotated output video, and export CSV.
    Returns a typed AnalysisResult — no positional tuple unpacking required.
    """
    model = model or YOLO(MODEL_PATH)

    with open_video(video_path) as cap:
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame    = end_frame if end_frame > 0 else total_frames

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        p1_positions:  list[PlayerPosition]  = []
        p2_positions:  list[PlayerPosition]  = []
        p1_trajectory: list[TrajectoryPoint] = []
        p2_trajectory: list[TrajectoryPoint] = []
        history:       deque                 = deque(maxlen=5)
        bg_frame    = None
        valid_count = 0
        frame_idx   = start_frame

        with open_writer("output_with_detections.mp4", 'mp4v', fps, (width, height)) as out, \
             open("player_positions.csv", "w", newline="") as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(["frame", "player1_x", "player1_y", "player2_x", "player2_y"])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx > end_frame:
                    break

                if bg_frame is None:
                    bg_frame = frame.copy()

                results = model.track(frame, persist=True, tracker="bytetrack.yaml")
                players = validate_and_extract(results, history, cfg)

                if players:
                    valid_count += 1
                    out.write(results[0].plot())
                    p1_det, p2_det = players
                    p1 = PlayerPosition(int(p1_det.cx), int(p1_det.cy))
                    p2 = PlayerPosition(int(p2_det.cx), int(p2_det.cy))
                    p1_positions.append(p1)
                    p2_positions.append(p2)
                    history.append((p1, p2))
                    p1_trajectory.append(TrajectoryPoint(frame_idx, p1.x, p1.y))
                    p2_trajectory.append(TrajectoryPoint(frame_idx, p2.x, p2.y))
                    writer.writerow([frame_idx + 1, p1.x, p1.y, p2.x, p2.y])
                else:
                    writer.writerow([frame_idx + 1, "", "", "", ""])

                frame_idx += 1
                if (frame_idx - start_frame) % 50 == 0:
                    logger.info("  %d/%d frames — %d valid",
                                frame_idx - start_frame,
                                end_frame - start_frame, valid_count)

    logger.info("Detection video → output_with_detections.mp4")
    logger.info("Positions CSV  → player_positions.csv")

    if bg_frame is not None:
        cv2.imwrite("background.jpg", bg_frame)

    return AnalysisResult(
        p1_positions=p1_positions,
        p2_positions=p2_positions,
        p1_trajectory=p1_trajectory,
        p2_trajectory=p2_trajectory,
        background_frame=bg_frame,
        width=width,
        height=height,
    )


# ── Visualisation helpers ──────────────────────────────────────────────────────

def create_enhanced_heatmap(result: AnalysisResult,
                             cfg: AnalyzerConfig = AnalyzerConfig()) -> None:
    """Render a high-resolution static heatmap overlaid on the background frame."""
    if not result.p1_positions or not result.p2_positions:
        logger.warning("No player positions — heatmap skipped")
        return
    if result.background_frame is None:
        logger.warning("No background frame — heatmap skipped")
        return

    heatmap = HeatmapAccumulator(result.width, result.height,
                                  sigma=cfg.heatmap_sigma_static,
                                  threshold=cfg.heatmap_threshold)
    for p1, p2 in zip(result.p1_positions, result.p2_positions):
        heatmap.add(p1, p2)

    combined       = heatmap.combined_density(sigma=cfg.heatmap_sigma_static)
    masked_combined = np.ma.array(combined, mask=(combined < cfg.heatmap_threshold))

    plt.figure(figsize=(20, 12), dpi=300)
    plt.imshow(cv2.cvtColor(result.background_frame, cv2.COLOR_BGR2RGB))
    heat_layer = plt.imshow(masked_combined, cmap=_STATIC_CMAP,
                             alpha=0.6, vmin=0, vmax=10)
    plt.title("Combined Player Position Heatmap", fontsize=24)
    plt.xlabel("X Position", fontsize=16)
    plt.ylabel("Y Position", fontsize=16)
    plt.colorbar(heat_layer).set_label("Density", fontsize=16)
    plt.tight_layout()
    plt.savefig("player_position_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Heatmap saved → player_position_heatmap.png")


def plot_player_movements(p1_trajectory: list[TrajectoryPoint],
                           p2_trajectory: list[TrajectoryPoint],
                           save_path: str = "player_movement_plot.png") -> None:
    """Plot X and Y positions of both players over time."""
    if not p1_trajectory or not p2_trajectory:
        logger.warning("No trajectory data to plot")
        return

    frames_p1 = [t.frame for t in p1_trajectory]
    frames_p2 = [t.frame for t in p2_trajectory]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(frames_p1, [t.x for t in p1_trajectory], label="Player 1 X", color='blue')
    plt.plot(frames_p2, [t.x for t in p2_trajectory], label="Player 2 X", color='red')
    plt.ylabel("X Position")
    plt.title("Player Movement Analysis")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(frames_p1, [t.y for t in p1_trajectory], label="Player 1 Y", color='blue')
    plt.plot(frames_p2, [t.y for t in p2_trajectory], label="Player 2 Y", color='red')
    plt.ylabel("Y Position")
    plt.xlabel("Frame")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Movement plot saved → %s", save_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Table Tennis Player Position Analysis")
    parser.add_argument("video_path")
    parser.add_argument("--start",      type=float, default=None,
                        help="Start time in seconds")
    parser.add_argument("--end",        type=float, default=None,
                        help="End time in seconds")
    parser.add_argument("--mode",       choices=["analyze", "heatmap", "video"],
                        default="analyze")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    # Logging configured here, not at module level
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Convert seconds → frames using the video's actual FPS.
    _cap   = cv2.VideoCapture(args.video_path)
    _fps   = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    start_frame = int(args.start * _fps) if args.start is not None else START_FRAME
    end_frame   = int(args.end   * _fps) if args.end   is not None else END_FRAME

    logger.info("Video: %s  |  frames: %d–%d  |  mode: %s",
                args.video_path, start_frame, end_frame, args.mode)

    # Model created once and passed into every function that needs it
    model = YOLO(MODEL_PATH)
    cfg   = AnalyzerConfig()

    if args.mode == "analyze":
        result = analyze_video(args.video_path, start_frame, end_frame, model, cfg)
        plot_player_movements(result.p1_trajectory, result.p2_trajectory)
        if result.p1_positions:
            logger.info("P1: %d positions | P2: %d positions",
                        len(result.p1_positions), len(result.p2_positions))
            create_enhanced_heatmap(result, cfg)
        else:
            logger.error("No valid player positions detected")

    elif args.mode == "heatmap":
        p1_pos, p2_pos, indices, fps, w, h = collect_valid_frames(
            args.video_path, start_frame, end_frame, model, cfg)
        if indices:
            # Build a minimal AnalysisResult so create_enhanced_heatmap can be reused
            with open_video(args.video_path) as cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
                _, bg = cap.read()
            result = AnalysisResult(p1_pos, p2_pos, [], [], bg, w, h)
            create_enhanced_heatmap(result, cfg)
        else:
            logger.error("No valid frames found")

    elif args.mode == "video":
        # Jump check disabled for heatmap video — fast in-rally movement between
        # frames is legitimate and should not be filtered out here.
        p1_pos, p2_pos, indices, fps, w, h = collect_valid_frames(
            args.video_path, start_frame, end_frame, model, cfg, check_jump=False)
        if len(indices) > 100:
            render_heatmap_video(args.video_path, indices, p1_pos, p2_pos,
                                  fps, w, h, frame_skip=args.frame_skip)
        elif indices:
            logger.warning("Too few valid frames (%d) — falling back to static heatmap",
                           len(indices))
            with open_video(args.video_path) as cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
                _, bg = cap.read()
            result = AnalysisResult(p1_pos, p2_pos, [], [], bg, w, h)
            create_enhanced_heatmap(result, cfg)
        else:
            logger.error("No valid frames found")

    logger.info("Done.")


if __name__ == "__main__":
    main()
