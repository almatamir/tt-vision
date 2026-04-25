from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from ultralytics import YOLO

from models import AnalyzerConfig
from table_tennis_analyzer import is_valid_frame

logger = logging.getLogger(__name__)

SHOT_TYPES: dict[str, int] = {
    'FOREHAND': 0,
    'BACKHAND': 1,
    'SMASH':    2,
    'PUSH':     3,
    'SERVE':    4,
    'UNKNOWN':  5,
}
SHOT_NAMES: dict[int, str] = {v: k for k, v in SHOT_TYPES.items()}


class ShotTypeClassifier:

    # COCO keypoint indices
    _NOSE           = 0
    _LEFT_SHOULDER  = 5
    _RIGHT_SHOULDER = 6
    _LEFT_ELBOW     = 7
    _RIGHT_ELBOW    = 8
    _LEFT_WRIST     = 9
    _RIGHT_WRIST    = 10
    _LEFT_HIP       = 11
    _RIGHT_HIP      = 12

    _SPEED_THRESHOLD    = 15.0
    _SMASH_SPEED        = 22.0
    _SMASH_WRIST_LIFT   = 20
    _BACKHAND_SPEED     = 10.0
    _FOREHAND_SPEED     = 12.0
    _PUSH_SPEED_MIN     = 5.0
    _PUSH_SPEED_MAX     = 12.0
    _MAX_FRAME_GAP      = 15
    _DEBOUNCE_FRAMES    = 5
    _HISTORY_TTL_FRAMES = 150
    _MIN_KPT_CONF       = 0.4
    _VOTE_WINDOW        = 5

    def __init__(self, model_path: str = "yolov8n-pose.pt",
                 cfg: AnalyzerConfig = AnalyzerConfig()) -> None:
        self.model = YOLO(model_path)
        self.cfg   = cfg
        self._movement_history: defaultdict[int, deque] = \
            defaultdict(lambda: deque(maxlen=10))
        self._last_seen: dict[int, int] = {}
        self._last_shot_info:  dict[int, dict] = {}
        self.shots_detected:   list[dict]       = []

    def analyze_video(self, video_path: str = "score_video.mp4",
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      output_dir: str = "output_shot",
                      sample_rate: int = 2,
                      visualize: bool = True,
                      slow_motion_factor: int = 1) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        try:
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame    = min(end_frame or total_frames, total_frames)

            slow_tag   = f"_slowmo{slow_motion_factor}x" if slow_motion_factor > 1 else ""
            out_path   = f"{output_dir}/shot_classification{slow_tag}.mp4"
            writer_fps = max(1.0, fps / sample_rate)
            out        = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         writer_fps, (width, height)) if visualize else None
            if out and not out.isOpened():
                out = None

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            shot_counters = [{v: 0 for v in SHOT_TYPES.values()} for _ in range(2)]
            frame_idx     = start_frame
            PERSISTENCE   = max(5, int(fps / sample_rate / 2))

            while cap.isOpened() and frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % sample_rate == 0:
                    self._prune_stale_ids(frame_idx)

                    results        = None
                    frame_is_valid = False
                    current_info   = {}

                    try:
                        results = self.model.track(frame, persist=True,
                                                    tracker="bytetrack.yaml",
                                                    verbose=False)
                        frame_is_valid = is_valid_frame(frame, results, cfg=self.cfg)
                    except Exception as e:
                        logger.warning("Frame %d — processing error: %s", frame_idx, e)

                    if (frame_is_valid and results and
                            results[0].boxes is not None and
                            results[0].keypoints is not None):
                        current_info = self._process_valid_frame(
                            frame, frame_idx, fps, results, shot_counters)

                    if visualize and out:
                        vis = self._draw_frame(
                            frame.copy(), frame_idx, frame_is_valid,
                            current_info, PERSISTENCE)
                        if frame_is_valid:
                            for _ in range(slow_motion_factor):
                                out.write(vis)

                frame_idx += 1

        finally:
            cap.release()
            if out:
                out.release()

        self._save_statistics(shot_counters, fps, output_dir)
        logger.info("Shot analysis complete — %d shots detected", len(self.shots_detected))
        return self.shots_detected

    def _prune_stale_ids(self, frame_idx: int) -> None:
        stale = [pid for pid, last in self._last_seen.items()
                 if frame_idx - last > self._HISTORY_TTL_FRAMES]
        for pid in stale:
            self._movement_history.pop(pid, None)
            self._last_seen.pop(pid, None)

    def _identify_players(self, detections_by_id: dict,
                           _width: int, _height: int) -> dict:
        candidates = []
        for id_, data in detections_by_id.items():
            if data['cls'] != 0:
                continue
            bbox     = data['bbox']
            x1, y1, x2, y2 = bbox
            cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
            w, h     = x2 - x1, y2 - y1
            if (w > self.cfg.min_player_width and
                    h > self.cfg.min_player_height and
                    self.cfg.y_min < cy < self.cfg.y_max):
                candidates.append({
                    'id': id_, 'bbox': bbox,
                    'keypoints': data['keypoints'],
                    'kpt_conf':  data.get('kpt_conf'),
                    'cx': cx,
                })

        candidates.sort(key=lambda p: p['cx'])
        selected = [candidates[0], candidates[-1]] if len(candidates) >= 2 \
                   else candidates[:1]
        return {p['id']: {'bbox': p['bbox'], 'keypoints': p['keypoints'],
                           'kpt_conf': p.get('kpt_conf')}
                for p in selected}

    def _process_valid_frame(self, frame, frame_idx: int, fps: float,
                              results, shot_counters: list) -> dict:
        kpts      = results[0].keypoints.xy.cpu().numpy()
        kpt_confs = (results[0].keypoints.conf.cpu().numpy()
                     if results[0].keypoints.conf is not None else None)
        boxes = results[0].boxes
        ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        current_info = {}

        if ids is None:
            return current_info

        detections_by_id = {
            int(id_): {
                'bbox':      boxes.xyxy.cpu().numpy()[i],
                'keypoints': kpts[i],
                'kpt_conf':  kpt_confs[i] if kpt_confs is not None else None,
                'cls':       boxes.cls.cpu().numpy()[i],
            }
            for i, id_ in enumerate(ids)
        }
        players = self._identify_players(
            detections_by_id, frame.shape[1], frame.shape[0])

        for player_idx, (player_id, data) in enumerate(players.items()):
            # Use court position (0=left, 1=right) as stable key instead of
            # ByteTrack's numeric ID, which resets on re-detection
            stable_id = player_idx
            self._last_seen[stable_id] = frame_idx
            bbox      = data['bbox']
            keypoints = data['keypoints']
            kpt_conf  = data['kpt_conf']
            current_info[stable_id] = {'bbox': bbox}

            shot_detected, shot_type = self._detect_shot(
                stable_id, keypoints, kpt_conf, frame_idx)

            if shot_detected:
                shot_name = SHOT_NAMES[shot_type]
                self.shots_detected.append({
                    'frame':               frame_idx,
                    'time':                frame_idx / fps,
                    'player_idx_relative': stable_id,
                    'shot_type':           shot_type,
                    'shot_name':           shot_name,
                    'cx':                  float((bbox[0] + bbox[2]) / 2),
                    'cy':                  float((bbox[1] + bbox[3]) / 2),
                })
                self._last_shot_info[stable_id] = {
                    'type': shot_type, 'name': shot_name, 'frame': frame_idx}
                if stable_id < len(shot_counters):
                    shot_counters[stable_id][shot_type] += 1

        return current_info

    def _draw_frame(self, vis: np.ndarray, frame_idx: int,
                    frame_is_valid: bool, current_info: dict,
                    persistence: int) -> np.ndarray:
        for stable_id, info in current_info.items():
            x1, y1, x2, y2 = map(int, info['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"P{stable_id + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for player_id, shot in list(self._last_shot_info.items()):
            age = frame_idx - shot['frame']
            if age >= persistence:
                self._last_shot_info.pop(player_id, None)
                continue
            if player_id in current_info:
                bbox = current_info[player_id]['bbox']
                cx   = int((bbox[0] + bbox[2]) / 2)
                y1   = int(bbox[1])
                cv2.putText(vis, shot['name'], (cx - 40, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                            cv2.LINE_AA)

        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        color = (0, 255, 0) if frame_is_valid else (0, 0, 255)
        cv2.putText(vis, f"Valid: {frame_is_valid}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    def _detect_shot(self, player_id: int, keypoints: np.ndarray,
                     kpt_conf: Optional[np.ndarray],
                     frame_idx: int) -> tuple[bool, int]:
        history = self._movement_history[player_id]

        joints = {
            'left_shoulder':  keypoints[self._LEFT_SHOULDER],
            'right_shoulder': keypoints[self._RIGHT_SHOULDER],
            'left_elbow':     keypoints[self._LEFT_ELBOW],
            'right_elbow':    keypoints[self._RIGHT_ELBOW],
            'left_wrist':     keypoints[self._LEFT_WRIST],
            'right_wrist':    keypoints[self._RIGHT_WRIST],
            'nose':           keypoints[self._NOSE],
            'left_hip':       keypoints[self._LEFT_HIP],
            'right_hip':      keypoints[self._RIGHT_HIP],
        }

        # Skip frames where YOLO isn't confident about the key joints
        key_indices = (self._LEFT_WRIST, self._RIGHT_WRIST,
                       self._LEFT_ELBOW, self._RIGHT_ELBOW)
        if (kpt_conf is not None and
                any(kpt_conf[i] < self._MIN_KPT_CONF for i in key_indices)):
            history.append({'frame': frame_idx, 'joints': joints,
                            'speed': 0.0, 'cls': SHOT_TYPES['UNKNOWN']})
            return False, SHOT_TYPES['UNKNOWN']

        # Compute wrist speed
        speed      = 0.0
        active_arm = 'right'
        if history:
            prev = history[-1]
            if frame_idx - prev['frame'] <= self._MAX_FRAME_GAP:
                pj = prev['joints']

                def _spd(side: str) -> float:
                    c, p = joints[f'{side}_wrist'], pj[f'{side}_wrist']
                    return (float(np.linalg.norm(c - p))
                            if not (np.isnan(c).any() or np.isnan(p).any())
                            else 0.0)

                left_spd, right_spd = _spd('left'), _spd('right')
                active_arm = 'left' if left_spd > right_spd else 'right'
                speed      = max(left_spd, right_spd)

        cls = (self._classify(active_arm, joints, speed)
               if speed > self._SPEED_THRESHOLD else SHOT_TYPES['UNKNOWN'])

        history.append({'frame': frame_idx, 'joints': joints,
                        'speed': speed, 'cls': cls})

        if len(history) < 3:
            return False, SHOT_TYPES['UNKNOWN']

        # Fire only when wrist speed just peaked (avoids duplicate triggers per swing)
        recent     = list(history)[-3:]
        spds       = [h['speed'] for h in recent]
        peak_speed = spds[1]
        if not (peak_speed > spds[0] and
                peak_speed > spds[2] and
                peak_speed > self._SPEED_THRESHOLD):
            return False, SHOT_TYPES['UNKNOWN']

        # Majority vote over recent above-threshold frames
        window = [h['cls'] for h in history
                  if h['speed'] > self._SPEED_THRESHOLD][-self._VOTE_WINDOW:]
        if not window:
            return False, SHOT_TYPES['UNKNOWN']

        voted_cls = Counter(window).most_common(1)[0][0]

        last = self._last_shot_info.get(player_id)
        if last and frame_idx - last['frame'] < self._DEBOUNCE_FRAMES:
            return False, voted_cls

        return True, voted_cls

    def _classify(self, active_arm: str, joints: dict, speed: float) -> int:
        shoulder = joints[f'{active_arm}_shoulder']
        elbow    = joints[f'{active_arm}_elbow']
        wrist    = joints[f'{active_arm}_wrist']
        opp_sh   = joints['right_shoulder' if active_arm == 'left' else 'left_shoulder']

        if any(np.isnan(j).any() for j in (shoulder, elbow, wrist, opp_sh)):
            return SHOT_TYPES['UNKNOWN']
        if np.linalg.norm(wrist - elbow) == 0:
            return SHOT_TYPES['UNKNOWN']

        body_cx      = (shoulder[0] + opp_sh[0]) / 2
        arm_crossing = (active_arm == 'right' and wrist[0] < body_cx) or \
                       (active_arm == 'left'  and wrist[0] > body_cx)

        if speed > self._SMASH_SPEED and wrist[1] < shoulder[1] - self._SMASH_WRIST_LIFT:
            return SHOT_TYPES['SMASH']
        elif arm_crossing and speed > self._BACKHAND_SPEED:
            return SHOT_TYPES['BACKHAND']
        elif not arm_crossing and speed > self._FOREHAND_SPEED:
            return SHOT_TYPES['FOREHAND']
        elif self._PUSH_SPEED_MIN < speed <= self._PUSH_SPEED_MAX:
            return SHOT_TYPES['PUSH']

        return SHOT_TYPES['UNKNOWN']

    def _save_statistics(self, shot_counters: list, _fps: float,
                          output_dir: str) -> None:
        plot_types  = [k for k in SHOT_TYPES if k != 'UNKNOWN']
        plot_values = [SHOT_TYPES[k] for k in plot_types]
        x, bar_w    = np.arange(len(plot_types)), 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x - bar_w / 2, [shot_counters[0].get(v, 0) for v in plot_values],
                bar_w, label='Player 1')
        plt.bar(x + bar_w / 2, [shot_counters[1].get(v, 0) for v in plot_values],
                bar_w, label='Player 2')
        plt.xlabel('Shot Type')
        plt.ylabel('Count')
        plt.title('Shot Type Distribution by Player')
        plt.xticks(x, plot_types, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shot_distribution.png", dpi=300)
        plt.close()

        if not self.shots_detected:
            return

        df       = pd.DataFrame(self.shots_detected)
        csv_path = f"{output_dir}/shot_data.csv"
        df.to_csv(csv_path, index=False)

        colors   = plt.colormaps['tab10'].resampled(len(plot_types))
        detected = df['player_idx_relative'].unique()

        plt.figure(figsize=(15, 6))
        for i, shot_name in enumerate(plot_types):
            color = colors(i)
            val   = SHOT_TYPES[shot_name]
            for player_idx, y_val in ((0, 1), (1, 2)):
                if player_idx not in detected:
                    continue
                pts = df[(df['player_idx_relative'] == player_idx) &
                          (df['shot_type'] == val)]
                if not pts.empty:
                    marker = '^' if player_idx == 0 else 'o'
                    plt.scatter(pts['time'], np.full(len(pts), y_val),
                                marker=marker, color=color, s=80)

        legend_elements = [
            Line2D([0], [0], marker='^', color='gray', label='Player 1',
                   markersize=8, linestyle='None'),
            Line2D([0], [0], marker='o', color='gray', label='Player 2',
                   markersize=8, linestyle='None'),
        ] + [Line2D([0], [0], color=colors(i), lw=4, label=stype)
             for i, stype in enumerate(plot_types)]

        plt.yticks([1, 2], ['Player 1', 'Player 2'])
        plt.xlabel('Time (seconds)')
        plt.title('Shot Sequence Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=max(3, len(plot_types)))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shot_sequence.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("===== SHOT SUMMARY =====")
        logger.info("Total: %d shots", len(self.shots_detected))
        for idx, label in enumerate(('Player 1', 'Player 2')):
            dist = {SHOT_NAMES[v]: c
                    for v, c in shot_counters[idx].items() if c > 0}
            if dist:
                logger.info("%s: %s", label,
                            ", ".join(f"{n}={c}" for n, c in
                                      sorted(dist.items(), key=lambda x: -x[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Table Tennis Shot Classifier")
    parser.add_argument("video_path")
    parser.add_argument("--start",       type=float, default=None)
    parser.add_argument("--end",         type=float, default=None)
    parser.add_argument("--sample-rate", type=int,   default=2)
    parser.add_argument("--output-dir",  default="output_shot")
    parser.add_argument("--no-vis",      action="store_true")
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    _cap   = cv2.VideoCapture(args.video_path)
    _fps   = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    start_frame = int(args.start * _fps) if args.start is not None else 0
    end_frame   = int(args.end   * _fps) if args.end   is not None else _total

    os.makedirs(args.output_dir, exist_ok=True)
    ShotTypeClassifier().analyze_video(
        args.video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        visualize=not args.no_vis,
    )


if __name__ == "__main__":
    main()
