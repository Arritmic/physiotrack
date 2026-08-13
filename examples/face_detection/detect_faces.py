#!/usr/bin/env python3
"""Detect faces in the bundled example scenes and save inspectable results.

Run this file without arguments from any directory.  Use ``--input`` to process a
different image or a directory tree containing images.
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


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXAMPLE_DIR / "data"
DEFAULT_OUTPUT = EXAMPLE_DIR / "results"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FACE_MODELS = {
    "nano": pt.Models.Detection.YOLO.FACE.n_face,
    "medium": pt.Models.Detection.YOLO.FACE.m_face,
    "large": pt.Models.Detection.YOLO.FACE.l_face,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One image or a directory searched recursively (default: bundled data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for annotated images, JSON, CSV, and run metadata.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(FACE_MODELS),
        default="medium",
        help="YOLO face model size (default: medium).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured runs on the first image before timing (default: 1).",
    )
    return parser.parse_args()


def image_paths(input_path: Path) -> tuple[Path, list[Path]]:
    """Return a common input root and a deterministic list of image paths."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    paths = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found below: {input_path}")
    return input_path, paths


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def uses_cuda(device: str) -> bool:
    token = str(device).lower()
    return token == "cuda" or token.startswith("cuda:") or token.isdigit()


def synchronize(device: str) -> None:
    if uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_predict(detector: Face, image: np.ndarray, device: str):
    synchronize(device)
    started = time.perf_counter()
    result = detector.predict(image)
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


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
    if args.warmup < 0:
        raise SystemExit("--warmup must be zero or greater")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise SystemExit("--conf and --iou must be between 0 and 1")

    try:
        input_root, paths = image_paths(args.input)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = FACE_MODELS[args.model]
    model_name = model.value

    setup_started = time.perf_counter()
    detector = pt.Face(
        model=model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    setup_ms = (time.perf_counter() - setup_started) * 1000.0

    first_image = load_image(paths[0])
    for _ in range(args.warmup):
        detector.predict(first_image)

    rows: list[dict] = []
    for image_path in paths:
        relative = image_path.relative_to(input_root)
        scene = relative.parts[0] if len(relative.parts) > 1 else "custom"
        annotated_path = output_dir / "annotated" / relative.with_suffix(".png")
        prediction_path = output_dir / "predictions" / relative.with_suffix(".json")
        try:
            image = load_image(image_path)
            result, inference_ms = timed_predict(detector, image, args.device)
            confidences = [
                float(instance.confidence)
                for instance in result
                if instance.confidence is not None
            ]

            annotated = result.plot(conf=True, color=(0, 220, 0), thickness=2)
            annotated = add_info_panel(
                annotated,
                [
                    f"Faces detected: {len(result)}",
                    f"Detector: {model_name}",
                    f"Device: {args.device} | {inference_ms:.1f} ms",
                ],
            )
            save_image(annotated_path, annotated)
            write_json(
                prediction_path,
                {
                    "source": {
                        "image": relative.as_posix(),
                        "scene": scene,
                        "width": image.shape[1],
                        "height": image.shape[0],
                    },
                    "configuration": {
                        "entry_point": "physiotrack.Face",
                        "model": model_name,
                        "device_requested": str(args.device),
                        "confidence_threshold": args.conf,
                        "nms_iou_threshold": args.iou,
                    },
                    "timing": {"inference_ms": inference_ms},
                    "result": result.to_dict(),
                },
            )
            row = {
                "image": relative.as_posix(),
                "scene": scene,
                "width": image.shape[1],
                "height": image.shape[0],
                "faces_detected": len(result),
                "mean_confidence": round(float(np.mean(confidences)), 6)
                if confidences
                else "",
                "minimum_confidence": round(min(confidences), 6) if confidences else "",
                "inference_ms": round(inference_ms, 3),
                "model": model_name,
                "device_requested": str(args.device),
                "status": "ok",
                "error": "",
            }
            print(f"{relative}: {len(result)} face(s), {inference_ms:.1f} ms")
        except Exception as exc:  # keep the rest of a collection usable
            row = {
                "image": relative.as_posix(),
                "scene": scene,
                "width": "",
                "height": "",
                "faces_detected": "",
                "mean_confidence": "",
                "minimum_confidence": "",
                "inference_ms": "",
                "model": model_name,
                "device_requested": str(args.device),
                "status": "error",
                "error": repr(exc),
            }
            print(f"{relative}: ERROR: {exc}")
        rows.append(row)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "ok"]
    write_json(
        output_dir / "run.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input),
            "output_dir": str(output_dir),
            "configuration": {
                "entry_point": "physiotrack.Face",
                "model": model_name,
                "device_requested": str(args.device),
                "confidence_threshold": args.conf,
                "nms_iou_threshold": args.iou,
                "warmup_runs": args.warmup,
            },
            "model_setup_ms": setup_ms,
            "images_found": len(rows),
            "images_succeeded": len(successful),
            "images_failed": len(rows) - len(successful),
            "total_faces_detected": sum(int(row["faces_detected"]) for row in successful),
            "mean_inference_ms": float(
                np.mean([float(row["inference_ms"]) for row in successful])
            )
            if successful
            else None,
            "runtime": runtime_info(),
            "important": (
                "These bundled synthetic scenes have no ground-truth boxes; the counts "
                "are qualitative example outputs, not detector-accuracy measurements."
            ),
        },
    )

    print(f"\nAnnotated images: {output_dir / 'annotated'}")
    print(f"Per-image JSON:   {output_dir / 'predictions'}")
    print(f"CSV summary:      {csv_path}")
    print(f"Run metadata:     {output_dir / 'run.json'}")
    if len(successful) != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
