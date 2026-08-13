#!/usr/bin/env python3
"""Compare VR-head, VR-person, and generic-person detection on one image."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import physiotrack as pt
import torch
from physiotrack.core.overlay import draw_info_panel


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR.parent / "face_detection" / "data" / "vr" / "vr_training_lab.jpg"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
DETECTORS = {
    "vr_head": {
        "factory": pt.Detection.VR,
        "question": "Where are the VR headsets?",
        "color": (255, 170, 0),
    },
    "vr_person": {
        "factory": pt.Detection.VRStudent,
        "question": "Which full people are using VR?",
        "color": (190, 0, 255),
    },
    "person": {
        "factory": pt.Detection.Person,
        "question": "Where are all people, with or without VR?",
        "color": (0, 210, 0),
    },
}
MODEL_PRESETS = {
    "medium": {
        "vr_head": pt.Models.Detection.YOLO.VR.m_vr,
        "vr_person": pt.Models.Detection.YOLO.VRSTUDENT.m_vrstudent,
        "person": pt.Models.Detection.YOLO.PERSON.m_person,
    },
    "large": {
        # No large VR-head checkpoint is currently published.
        "vr_head": None,
        "vr_person": pt.Models.Detection.YOLO.VRSTUDENT.l_vrstudent,
        "person": pt.Models.Detection.YOLO.PERSON.l_person,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for annotated images and summary.json.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=tuple(DETECTORS),
        default=list(DETECTORS),
        help="Detector views to run (default: all three).",
    )
    parser.add_argument(
        "--model-size",
        choices=("medium", "large", "largest"),
        default="medium",
        help=(
            "Checkpoint size. 'large' is strict; 'largest' uses the largest "
            "published model for each detector (VR-head is medium-only)."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def uses_cuda(device: str) -> bool:
    token = str(device).lower()
    return token == "cuda" or token.startswith("cuda:") or token.isdigit()


def synchronize(device: str) -> None:
    if uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def select_models(detectors: list[str], model_size: str) -> dict:
    """Select published checkpoints without silently substituting model sizes."""
    if model_size == "largest":
        return {
            key: MODEL_PRESETS["large"][key] or MODEL_PRESETS["medium"][key]
            for key in detectors
        }

    selected = {key: MODEL_PRESETS[model_size][key] for key in detectors}
    unavailable = [key for key, model in selected.items() if model is None]
    if unavailable:
        names = ", ".join(unavailable)
        raise ValueError(
            f"No {model_size} checkpoint is published for: {names}. "
            "Use --model-size largest to use the largest available checkpoint for "
            "each detector, or omit vr_head when requesting --model-size large."
        )
    return selected


def timed_predict(detector, image: np.ndarray, device: str):
    synchronize(device)
    started = time.perf_counter()
    result = detector.predict(image)
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def class_counts(result) -> dict[str, int]:
    """Count the actual class labels returned by a detector result."""
    counts = Counter(
        instance.cls_name or f"class_{instance.cls}"
        for instance in result
    )
    return dict(sorted(counts.items()))


def panel_lines(key: str, result, elapsed_ms: float, model_name: str, device: str) -> list[str]:
    """Build the explanatory panel shown on one detector view."""
    counts = class_counts(result)
    count_text = ", ".join(f"{label}: {count}" for label, count in counts.items())
    return [
        DETECTORS[key]["question"],
        count_text or "No detections",
        f"Detector: {model_name}",
        f"Device: {device} | {elapsed_ms:.1f} ms",
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def stack_views(views: list[np.ndarray], width: int = 900) -> np.ndarray:
    """Resize annotated views to one width and stack them vertically."""
    preview_width = min(width, min(view.shape[1] for view in views))
    resized = [
        cv2.resize(
            view,
            (preview_width, round(view.shape[0] * preview_width / view.shape[1])),
        )
        for view in views
    ]
    return np.vstack(resized)


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--conf and --iou must be between 0 and 1")
    if uses_cuda(args.device) and not torch.cuda.is_available():
        raise SystemExit(f"Device {args.device!r} requests CUDA, but CUDA is unavailable")

    input_path = args.input.expanduser().resolve()
    image = cv2.imread(str(input_path))
    if image is None:
        raise SystemExit(f"OpenCV could not decode input image: {input_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        selected_models = select_models(args.detectors, args.model_size)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    records = []
    annotated_views = []
    annotated_by_detector = {}
    for key in args.detectors:
        spec = DETECTORS[key]
        model = selected_models[key]
        detector = spec["factory"](
            model=model, conf=args.conf, iou=args.iou, device=args.device
        )
        for _ in range(args.warmup):
            detector.predict(image)
        result, elapsed_ms = timed_predict(detector, image, args.device)
        annotated = result.plot(conf=True, color=spec["color"], thickness=3)
        annotated = draw_info_panel(
            annotated,
            panel_lines(key, result, elapsed_ms, model.value, str(args.device)),
        )
        output_path = output_dir / f"{key}.png"
        save_image(output_path, annotated)
        annotated_views.append(annotated)
        annotated_by_detector[key] = annotated
        records.append({
            "detector": key,
            "entry_point": f"physiotrack.Detection.{spec['factory'].__name__}",
            "question": spec["question"],
            "model": model.value,
            "device_requested": str(args.device),
            "inference_ms": elapsed_ms,
            "detections": len(result),
            "class_counts": class_counts(result),
            "result": result.to_dict(),
            "annotated_image": output_path.name,
        })
        print(f"{key}: {len(result)} detection(s), {elapsed_ms:.1f} ms -> {output_path}")

    if len(annotated_views) > 1:
        save_image(output_dir / "comparison.png", stack_views(annotated_views))
    if {"vr_person", "person"} <= annotated_by_detector.keys():
        # This focused JPEG can be copied into docs/images as a compact preview.
        focused = [annotated_by_detector["vr_person"], annotated_by_detector["person"]]
        save_image(
            output_dir / "comparison_person_vrperson.jpg",
            stack_views(focused),
        )

    write_json(output_dir / "summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "configuration": {
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "warmup_runs_per_detector": args.warmup,
            "model_size_policy": args.model_size,
        },
        "detectors": records,
        "important": (
            "The three counts are not expected to match: the models detect different "
            "regions or subject categories. This synthetic image has no ground-truth "
            "annotations, so these are qualitative outputs rather than accuracy metrics."
        ),
    })
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
