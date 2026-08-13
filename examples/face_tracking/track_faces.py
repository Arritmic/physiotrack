#!/usr/bin/env python3
"""Detect and track faces in the bundled video, then save visual and tabular output.

Track IDs are temporary associations within one video.  This example performs no face
recognition and does not determine anyone's identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import physiotrack as pt
import torch
from physiotrack.face import Face
from physiotrack.results import Result


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data" / "students_face_tracking.mp4"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
FACE_MODELS = {
    "nano": pt.Models.Detection.YOLO.FACE.n_face,
    "medium": pt.Models.Detection.YOLO.FACE.m_face,
    "large": pt.Models.Detection.YOLO.FACE.l_face,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the annotated video, JSONL, CSV, and run summary.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(FACE_MODELS),
        default="medium",
        help="YOLO face model size (default: medium).",
    )
    parser.add_argument("--device", default="cpu", help="Face detector device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--tracker",
        choices=("ocsort", "bytetrack", "boosttrack"),
        default="ocsort",
        help="Tracking-by-detection backend (default: ocsort).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames for a quick smoke run.",
    )
    parser.add_argument("--show", action="store_true", help="Show a live window; press q to stop.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def uses_cuda(device: str) -> bool:
    token = str(device).lower()
    return token == "cuda" or token.startswith("cuda:") or token.isdigit()


def synchronize(device: str) -> None:
    if uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_predict(detector: Face, frame: np.ndarray, device: str):
    synchronize(device)
    started = time.perf_counter()
    result = detector.predict(frame)
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def result_to_tracker_rows(result: Result) -> np.ndarray:
    """Convert a Result into tracker rows [x1, y1, x2, y2, confidence, class]."""
    rows = [
        [
            *np.asarray(instance.box, dtype=float),
            1.0 if instance.confidence is None else float(instance.confidence),
            0 if instance.cls is None else int(instance.cls),
        ]
        for instance in result
        if instance.box is not None
    ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def add_info_panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a readable, size-adaptive information panel in the top-left corner."""
    output = image.copy()
    height, width = output.shape[:2]
    font_scale = float(np.clip(min(width, height) / 1050.0, 0.55, 1.15))
    thickness = max(1, int(round(font_scale * 2)))
    margin = max(10, int(round(14 * font_scale)))
    line_gap = max(7, int(round(9 * font_scale)))
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in lines
    ]
    panel_width = min(width, max(size[0] for size in sizes) + 2 * margin)
    panel_height = min(
        height,
        sum(size[1] for size in sizes) + line_gap * (len(lines) - 1) + 2 * margin,
    )

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
    output[:panel_height, :panel_width] = cv2.addWeighted(
        overlay[:panel_height, :panel_width],
        0.78,
        output[:panel_height, :panel_width],
        0.22,
        0,
    )

    y = margin + sizes[0][1]
    for line, (_, text_height) in zip(lines, sizes):
        cv2.putText(
            output,
            line,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += text_height + line_gap
    return output


def runtime_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "physiotrack": pt.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--conf and --iou must be between 0 and 1")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input video does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps or not np.isfinite(fps):
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit(f"Video reports an invalid frame size: {width}x{height}")

    output_video = output_dir / f"{input_path.stem}_tracked.mp4"
    frame_records = output_dir / f"{input_path.stem}_frames.jsonl"
    tracks_csv = output_dir / f"{input_path.stem}_tracks.csv"
    summary_path = output_dir / f"{input_path.stem}_summary.json"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"OpenCV could not create output video: {output_video}")

    model = FACE_MODELS[args.model]
    model_name = model.value
    setup_started = time.perf_counter()
    detector = pt.Face(
        model=model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    detector_setup_ms = (time.perf_counter() - setup_started) * 1000.0
    tracker_config = pt.TrackerConfig(
        tracker_type=args.tracker,
        classes=[0],
        bytetrack_frame_rate=max(1, round(fps)),
        show_detection_boxes=False,
        show_original_tracks=True,
        show_all_trails=True,
        enable_subject_lock=False,
    )
    tracker = pt.Tracker(tracker_config)

    csv_fields = [
        "frame_index",
        "timestamp_seconds",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "class_id",
    ]
    frame_index = 0
    face_counts: list[int] = []
    detection_times: list[float] = []
    tracking_times: list[float] = []
    unique_ids: set[int] = set()
    processing_started = time.perf_counter()

    try:
        with (
            frame_records.open("w", encoding="utf-8") as jsonl,
            tracks_csv.open("w", newline="", encoding="utf-8") as csv_handle,
        ):
            csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fields)
            csv_writer.writeheader()
            while args.max_frames is None or frame_index < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                face_result, detection_ms = timed_predict(detector, frame, args.device)
                detections = result_to_tracker_rows(face_result)
                tracking_started = time.perf_counter()
                track_result = tracker.track(frame.copy(), detections)
                tracking_ms = (time.perf_counter() - tracking_started) * 1000.0

                active_ids = track_result.ids
                annotated = add_info_panel(
                    track_result.plot(),
                    [
                        f"Faces detected: {len(face_result)} | Active tracks: {len(track_result)}",
                        f"Detector: {model_name}",
                        f"Tracker: {args.tracker.upper()} | Frame: {frame_index}",
                    ],
                )
                writer.write(annotated)

                timestamp = frame_index / fps
                jsonl.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "faces_detected": len(face_result),
                            "active_track_ids": active_ids,
                            "face_result": face_result.to_dict(),
                            "track_result": track_result.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for instance in track_result:
                    if instance.box is None:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in instance.box)
                    csv_writer.writerow(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(timestamp, 6),
                            "track_id": instance.id,
                            "x1": round(x1, 3),
                            "y1": round(y1, 3),
                            "x2": round(x2, 3),
                            "y2": round(y2, 3),
                            "confidence": ""
                            if instance.confidence is None
                            else round(float(instance.confidence), 6),
                            "class_id": instance.cls,
                        }
                    )

                face_counts.append(len(face_result))
                detection_times.append(detection_ms)
                tracking_times.append(tracking_ms)
                unique_ids.update(active_ids)
                if frame_index % max(1, round(fps)) == 0:
                    print(
                        f"frame {frame_index}: {len(face_result)} face(s), "
                        f"active IDs={active_ids}"
                    )

                frame_index += 1
                if args.show:
                    cv2.imshow("PhysioTrack face tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - processing_started
    write_json(
        summary_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path),
            "outputs": {
                "annotated_video": str(output_video),
                "frame_records_jsonl": str(frame_records),
                "track_rows_csv": str(tracks_csv),
            },
            "source": {
                "width": width,
                "height": height,
                "fps": fps,
                "declared_frames": declared_frames,
                "audio_copied_to_output": False,
            },
            "configuration": {
                "entry_point": "physiotrack.Face",
                "face_model": model_name,
                "face_detector_device": str(args.device),
                "confidence_threshold": args.conf,
                "nms_iou_threshold": args.iou,
                "tracker": args.tracker,
                "max_frames": args.max_frames,
            },
            "detector_setup_ms": detector_setup_ms,
            "processed_frames": frame_index,
            "total_face_detections": sum(face_counts),
            "mean_faces_per_frame": float(np.mean(face_counts)) if face_counts else None,
            "maximum_faces_in_one_frame": max(face_counts) if face_counts else None,
            "unique_temporary_track_ids": sorted(unique_ids),
            "elapsed_seconds": elapsed,
            "end_to_end_fps_including_video_io": frame_index / elapsed if elapsed else None,
            "mean_detection_ms": float(np.mean(detection_times)) if detection_times else None,
            "mean_tracking_ms": float(np.mean(tracking_times)) if tracking_times else None,
            "runtime": runtime_info(),
            "important": (
                "Track IDs are temporary within-video associations, not recognized "
                "identities. The synthetic clip has no ground-truth tracks, so this is a "
                "qualitative example rather than a tracking-accuracy measurement."
            ),
        },
    )

    print(f"\nProcessed frames: {frame_index}")
    print(f"Temporary IDs observed: {sorted(unique_ids)}")
    print(f"Annotated video: {output_video}")
    print(f"Per-frame JSONL: {frame_records}")
    print(f"Track CSV:       {tracks_csv}")
    print(f"Run summary:     {summary_path}")


if __name__ == "__main__":
    main()
