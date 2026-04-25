from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from table_tennis_analyzer import open_video, open_writer

logger = logging.getLogger(__name__)

OUTPUT_VIDEO       = "score_video.mp4"
OUTPUT_SCORE_LOG   = "score_chart.csv"
OUTPUT_CHART       = "score_chart.png"
OUTPUT_SCREENSHOTS = "score_screenshots"

# Score region boundaries as fractions of frame size — adjust if broadcast layout changes
PLAYER1_Y_RANGE = (0.86, 0.90)
PLAYER2_Y_RANGE = (0.90, 0.95)
WINS_X_RANGE    = (0.23, 0.26)
POINTS_X_RANGE  = (0.26, 0.29)

# EasyOCR is initialised lazily to avoid the weight download cost on every import
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def _crop_score_region(frame: np.ndarray, player: int, stat: str) -> np.ndarray:
    y_range = PLAYER1_Y_RANGE if player == 1 else PLAYER2_Y_RANGE
    x_range = WINS_X_RANGE   if stat == "wins" else POINTS_X_RANGE
    h, w    = frame.shape[:2]
    x1, x2  = int(w * x_range[0]), int(w * x_range[1])
    y1, y2  = int(h * y_range[0]), int(h * y_range[1])
    return cv2.resize(frame[y1:y2, x1:x2], None, fx=2, fy=2,
                      interpolation=cv2.INTER_CUBIC)


def save_score_screenshot(
        frame:      np.ndarray,
        frame_idx:  int,
        p1_wins:    int,
        p1_points:  int,
        p2_wins:    int,
        p2_points:  int,
        output_dir: str,
) -> None:
    """Save a debug composite image whenever a score change is detected."""
    crop_h = 80
    crops = {
        f"P1 wins={p1_wins}":   _crop_score_region(frame, 1, "wins"),
        f"P1 pts={p1_points}":  _crop_score_region(frame, 1, "points"),
        f"P2 wins={p2_wins}":   _crop_score_region(frame, 2, "wins"),
        f"P2 pts={p2_points}":  _crop_score_region(frame, 2, "points"),
    }

    thumbs = []
    for label, crop in crops.items():
        h, w  = crop.shape[:2]
        scale = crop_h / h
        thumb = cv2.resize(crop, (max(1, int(w * scale)), crop_h))
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        canvas = np.zeros((crop_h + 20, thumb.shape[1], 3), dtype=np.uint8)
        cv2.putText(canvas, label, (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        canvas[20:] = thumb
        thumbs.append(canvas)

    max_w  = max(t.shape[1] for t in thumbs)
    padded = [np.pad(t, ((0, 0), (0, max_w - t.shape[1]), (0, 0))) for t in thumbs]
    strip  = np.hstack(padded)

    annotated = frame.copy()
    h, w = frame.shape[:2]
    y1 = int(h * PLAYER1_Y_RANGE[0])
    y2 = int(h * PLAYER2_Y_RANGE[1])
    x1 = int(w * WINS_X_RANGE[0])
    x2 = int(w * POINTS_X_RANGE[1])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated,
                f"Frame {frame_idx}  P1: {p1_wins}W {p1_points}pts  "
                f"P2: {p2_wins}W {p2_points}pts",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if strip.shape[1] != w:
        strip = cv2.resize(strip, (w, int(strip.shape[0] * w / strip.shape[1])))

    composite = np.vstack([annotated, strip])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = (f"frame_{frame_idx:06d}_P1-{p1_wins}W{p1_points}pts"
             f"_P2-{p2_wins}W{p2_points}pts.jpg")
    cv2.imwrite(str(Path(output_dir) / fname), composite,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def read_scores(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """OCR the scoreboard. Returns (p1_total, p1_points, p2_total, p2_points) or None."""
    reader = _get_reader()
    try:
        p1_wins   = int(reader.readtext(_crop_score_region(frame, 1, "wins"))[0][1])
        p2_wins   = int(reader.readtext(_crop_score_region(frame, 2, "wins"))[0][1])
        p1_points = int(reader.readtext(_crop_score_region(frame, 1, "points"))[0][1])
        p2_points = int(reader.readtext(_crop_score_region(frame, 2, "points"))[0][1])
        return p1_wins * 10 + p1_points, p1_points, p2_wins * 10 + p2_points, p2_points
    except (IndexError, ValueError):
        return None


def load_shot_data(csv_path: str) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Shot data not found at %s — shot overlay disabled", csv_path)
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['frame']               = int(r['frame'])
        r['player_idx_relative'] = int(r['player_idx_relative'])
    rows.sort(key=lambda r: r['frame'])
    return rows


def get_active_shots(shot_data: list[dict], frame_idx: int,
                     persist_frames: int = 45) -> dict[int, str]:
    active: dict[int, str] = {}
    for shot in reversed(shot_data):
        f = shot['frame']
        if f > frame_idx:
            continue
        if frame_idx - f > persist_frames:
            break
        pidx = int(shot['player_idx_relative'])
        if pidx not in active:
            active[pidx] = shot['shot_name']
    return active


def overlay_scores(frame: np.ndarray,
                   p1_points: int, p2_points: int,
                   p1_wins: int,   p2_wins: int,
                   p1_shot: str = "", p2_shot: str = "") -> np.ndarray:
    w       = frame.shape[1]
    panel_h = 120 if (p1_shot or p2_shot) else 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 320, 0), (w, panel_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Player 1: {p1_wins}W  {p1_points}pts",
                (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Player 2: {p2_wins}W  {p2_points}pts",
                (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if p1_shot or p2_shot:
        shot_text = f"P1: {p1_shot or '—'}   P2: {p2_shot or '—'}"
        cv2.putText(frame, shot_text,
                    (w - 300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    return frame


def _frame_to_time(frame: int, fps: float) -> str:
    seconds = frame / fps
    return f"{int(seconds // 60)}:{int(seconds % 60):02}"


def save_chart(score_list: list, fps: float, output_path: str) -> None:
    if not score_list:
        return

    frames  = [s[0] for s in score_list]
    s1_list = [s[1] for s in score_list]
    s2_list = [s[3] for s in score_list]

    plt.figure(figsize=(12, 6))
    plt.plot(frames, s1_list, 'b-', label="Player 1", linewidth=3, marker='o', markersize=6)
    plt.plot(frames, s2_list, 'r-', label="Player 2", linewidth=3, marker='s', markersize=6)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Score (10×Wins + Points)", fontsize=12)
    plt.title("Score Progression", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    if frames:
        tick_idx    = np.linspace(0, len(frames) - 1, min(len(frames), 10), dtype=int)
        tick_frames = [frames[i] for i in tick_idx]
        plt.xticks(tick_frames, [_frame_to_time(f, fps) for f in tick_frames], rotation=45)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, format='png')
    plt.close()


def save_csv(score_list: list, fps: float, output_path: str) -> None:
    if not score_list:
        return
    csv_path     = Path(output_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write("Frame,Time,Player1_Score,Player2_Score,"
                    "Player1_Games,Player1_Points,Player2_Games,Player2_Points\n")
        for frame, s1_total, s1_pts, s2_total, s2_pts in score_list:
            f.write(f"{frame},{_frame_to_time(frame, fps)},"
                    f"{s1_total},{s2_total},"
                    f"{s1_total // 10},{s1_pts},"
                    f"{s2_total // 10},{s2_pts}\n")


_ORDINALS = [
    "First", "Second", "Third", "Fourth", "Fifth",
    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
]

def _ordinal(n: int) -> str:
    if 1 <= n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    return f"{n}{suffix}"


def make_title_card(
        width: int, height: int,
        event_number: int,
        p1_wins: int, p1_points: int,
        p2_wins: int, p2_points: int,
        timestamp: str,
        scorer: str = "",
        shot_name: str = "",
) -> np.ndarray:
    card    = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (20, 20, 20)
    cx      = width // 2

    heading = f"{_ordinal(event_number)} Score"
    (tw, _), _ = cv2.getTextSize(heading, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
    cv2.putText(card, heading, (cx - tw // 2, height // 2 - 60),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

    score_line = (f"Player 1:  {p1_wins}W  {p1_points}pts"
                  f"    |    "
                  f"Player 2:  {p2_wins}W  {p2_points}pts")
    (tw, _), _ = cv2.getTextSize(score_line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(card, score_line, (cx - tw // 2, height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

    if scorer or shot_name:
        detail = "  |  ".join(filter(None, [scorer, shot_name]))
        (tw, _), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(card, detail, (cx - tw // 2, height // 2 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 100), 2, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(card, timestamp, (cx - tw // 2, height // 2 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Table Tennis Score Analyzer")
    parser.add_argument("video_path")
    parser.add_argument("--start",           type=float, default=1150)
    parser.add_argument("--end",             type=float, default=1190)
    parser.add_argument("--output-video",    default=OUTPUT_VIDEO)
    parser.add_argument("--output-chart",    default=OUTPUT_CHART)
    parser.add_argument("--output-csv",      default=OUTPUT_SCORE_LOG)
    parser.add_argument("--screenshots-dir", default=OUTPUT_SCREENSHOTS)
    parser.add_argument("--shot-data",       default="output_shot/shot_data.csv")
    parser.add_argument("--debug",           action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    screenshots_dir = Path(args.screenshots_dir)
    if screenshots_dir.exists():
        shutil.rmtree(screenshots_dir)
    screenshots_dir.mkdir(parents=True)

    shot_data = load_shot_data(args.shot_data)

    with open_video(args.video_path) as cap:
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(args.start * fps)
        end_frame   = int(args.end   * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        score_list  = []
        last_scores = None
        score_count = 0
        card_frames = max(1, int(fps * 1.5))
        s1 = s2 = s1_total = s2_total = 0
        frame_idx   = start_frame

        with open_writer(args.output_video, 'mp4v', fps, (width, height)) as out:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                active_shots = get_active_shots(shot_data, frame_idx)
                p1_shot = active_shots.get(0, "")
                p2_shot = active_shots.get(1, "")

                scores = read_scores(frame)
                if scores and scores != last_scores:
                    s1_total, s1, s2_total, s2 = scores
                    score_list.append((frame_idx, s1_total, s1, s2_total, s2))
                    last_scores = scores
                    score_count += 1
                    logger.info("Frame %d: P1=%d pts, P2=%d pts", frame_idx, s1, s2)
                    save_score_screenshot(
                        frame, frame_idx,
                        p1_wins=s1_total // 10, p1_points=s1,
                        p2_wins=s2_total // 10, p2_points=s2,
                        output_dir=args.screenshots_dir,
                    )
                    if score_list and len(score_list) >= 2:
                        prev   = score_list[-2]
                        scorer = ("Player 1 scores" if s1_total > prev[1]
                                  else "Player 2 scores")
                        last_shot = p1_shot if s1_total > prev[1] else p2_shot
                    else:
                        scorer, last_shot = "", ""

                    card = make_title_card(
                        width, height,
                        event_number=score_count,
                        p1_wins=s1_total // 10, p1_points=s1,
                        p2_wins=s2_total // 10, p2_points=s2,
                        timestamp=_frame_to_time(frame_idx, fps),
                        scorer=scorer,
                        shot_name=last_shot,
                    )
                    for _ in range(card_frames):
                        out.write(card)
                elif last_scores:
                    s1_total, s1, s2_total, s2 = last_scores

                out.write(overlay_scores(frame, s1, s2,
                                         s1_total // 10, s2_total // 10,
                                         p1_shot, p2_shot))
                frame_idx += 1

    logger.info("Recorded %d score events", len(score_list))
    save_chart(score_list, fps, args.output_chart)
    save_csv(score_list,   fps, args.output_csv)


if __name__ == "__main__":
    main()
