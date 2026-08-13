#!/usr/bin/env python3
"""Compare CPU and CUDA face detection on exactly the same image.

The benchmark warms up each device separately, measures repeated end-to-end
``Face.predict`` calls, and checks whether the final CPU/GPU boxes agree. It requires
a CUDA-enabled PyTorch installation and an NVIDIA GPU visible to this process.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import physiotrack as pt
import torch
from physiotrack.core.overlay import draw_info_panel


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data" / "pov" / "exercise_class_pov.jpg"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results" / "cpu_vs_gpu"
FACE_MODELS = {
    "nano": pt.Models.Detection.YOLO.FACE.n_face,
    "medium": pt.Models.Detection.YOLO.FACE.m_face,
    "large": pt.Models.Detection.YOLO.FACE.l_face,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for CPU/GPU images and comparison.json.",
    )
    parser.add_argument("--model", choices=tuple(FACE_MODELS), default="medium")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--match-iou", type=float, default=0.50,
        help="Minimum box IoU for a CPU/GPU prediction match (default: 0.50).",
    )
    return parser.parse_args()


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two ``(N, 4)`` arrays in xyxy coordinates."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=float)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[..., 0] * size[..., 1]
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def match_predictions(cpu_result, gpu_result, minimum_iou: float) -> dict:
    """Greedily match highest-IoU boxes and summarize CPU/GPU agreement."""
    matrix = box_iou_matrix(cpu_result.boxes, gpu_result.boxes)
    candidates = sorted(
        (
            (float(matrix[cpu_i, gpu_i]), cpu_i, gpu_i)
            for cpu_i in range(matrix.shape[0])
            for gpu_i in range(matrix.shape[1])
            if matrix[cpu_i, gpu_i] >= minimum_iou
        ),
        reverse=True,
    )
    used_cpu: set[int] = set()
    used_gpu: set[int] = set()
    matches = []
    for box_iou, cpu_i, gpu_i in candidates:
        if cpu_i in used_cpu or gpu_i in used_gpu:
            continue
        used_cpu.add(cpu_i)
        used_gpu.add(gpu_i)
        cpu_conf = float(cpu_result[cpu_i].confidence or 0.0)
        gpu_conf = float(gpu_result[gpu_i].confidence or 0.0)
        matches.append({
            "cpu_index": cpu_i,
            "gpu_index": gpu_i,
            "box_iou": box_iou,
            "absolute_confidence_difference": abs(cpu_conf - gpu_conf),
        })
    return {
        "cpu_detections": len(cpu_result),
        "gpu_detections": len(gpu_result),
        "matched_detections": len(matches),
        "unmatched_cpu": len(cpu_result) - len(matches),
        "unmatched_gpu": len(gpu_result) - len(matches),
        "mean_matched_iou": (
            float(np.mean([match["box_iou"] for match in matches])) if matches else None
        ),
        "matches": matches,
    }


def timed_predict(detector, image: np.ndarray, device: str):
    """Synchronize CUDA when needed and return ``(result, elapsed_ms)``."""
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = detector.predict(image)
    if device == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def benchmark(detector, image: np.ndarray, device: str, warmup: int, repeats: int):
    """Warm up one device and time repeated end-to-end predictions."""
    result = None
    for _ in range(warmup):
        result, _ = timed_predict(detector, image, device)
    values = []
    for _ in range(repeats):
        result, elapsed_ms = timed_predict(detector, image, device)
        values.append(elapsed_ms)
    return result, values


def timing_summary(values: list[float]) -> dict:
    """Return descriptive timing statistics and all raw measurements."""
    mean_ms = statistics.fmean(values)
    return {
        "repeats": len(values),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mean_fps": 1000.0 / mean_ms,
        "all_ms": values,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be at least 1 and --warmup cannot be negative")
    if not all(0.0 <= value <= 1.0 for value in (args.conf, args.iou, args.match_iou)):
        raise SystemExit("--conf, --iou, and --match-iou must be between 0 and 1")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this Python process. Install a CUDA-enabled "
            "PyTorch build and confirm torch.cuda.is_available() before benchmarking."
        )

    image_path = args.input.expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"OpenCV could not decode input image: {image_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = FACE_MODELS[args.model]

    cpu_detector = pt.Face(model=model, conf=args.conf, iou=args.iou, device="cpu")
    gpu_detector = pt.Face(model=model, conf=args.conf, iou=args.iou, device="cuda")
    cpu_result, cpu_values = benchmark(
        cpu_detector, image, "cpu", args.warmup, args.repeats
    )
    torch.cuda.reset_peak_memory_stats()
    gpu_result, gpu_values = benchmark(
        gpu_detector, image, "cuda", args.warmup, args.repeats
    )

    cpu_stats = timing_summary(cpu_values)
    gpu_stats = timing_summary(gpu_values)
    agreement = match_predictions(cpu_result, gpu_result, args.match_iou)
    speedup = cpu_stats["mean_ms"] / gpu_stats["mean_ms"]

    cpu_image = draw_info_panel(cpu_result.plot(conf=True), [
        f"Device: CPU | Faces: {len(cpu_result)}",
        f"Mean: {cpu_stats['mean_ms']:.1f} ms | {cpu_stats['mean_fps']:.1f} FPS",
        f"Detector: {model.value}",
    ])
    gpu_image = draw_info_panel(gpu_result.plot(conf=True), [
        f"Device: CUDA | Faces: {len(gpu_result)}",
        f"Mean: {gpu_stats['mean_ms']:.1f} ms | {gpu_stats['mean_fps']:.1f} FPS",
        f"Detector: {model.value}",
    ])
    save_image(output_dir / "cpu.png", cpu_image)
    save_image(output_dir / "gpu.png", gpu_image)
    side_by_side = np.hstack([cpu_image, gpu_image])
    save_image(output_dir / "side_by_side.png", side_by_side)
    # JPEG is the compact source that can be copied into docs/images.
    preview_height = min(1000, side_by_side.shape[0])
    preview_width = round(side_by_side.shape[1] * preview_height / side_by_side.shape[0])
    preview = cv2.resize(side_by_side, (preview_width, preview_height))
    save_image(output_dir / "side_by_side.jpg", preview)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(image_path),
        "configuration": {
            "entry_point": "physiotrack.Face",
            "model": model.value,
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "warmup_runs_per_device": args.warmup,
            "measured_repeats_per_device": args.repeats,
            "box_match_iou_threshold": args.match_iou,
        },
        "timing": {"cpu": cpu_stats, "gpu": gpu_stats},
        "gpu_speedup_over_cpu": speedup,
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "prediction_agreement": agreement,
        "runtime": {
            "physiotrack": pt.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "important": (
            "Timing is specific to this hardware, software stack, model, image, and "
            "benchmark configuration; it is not a universal CPU/GPU claim."
        ),
    }
    write_json(output_dir / "comparison.json", payload)

    print(f"CPU: {cpu_stats['mean_ms']:.1f} ms ({cpu_stats['mean_fps']:.1f} FPS)")
    print(f"GPU: {gpu_stats['mean_ms']:.1f} ms ({gpu_stats['mean_fps']:.1f} FPS)")
    print(f"GPU speed-up: {speedup:.2f}x")
    print(
        "Detections CPU/GPU/matched: "
        f"{len(cpu_result)}/{len(gpu_result)}/{agreement['matched_detections']}"
    )
    print(f"Report: {output_dir / 'comparison.json'}")
    print(f"Visual comparison: {output_dir / 'side_by_side.jpg'}")


if __name__ == "__main__":
    main()
