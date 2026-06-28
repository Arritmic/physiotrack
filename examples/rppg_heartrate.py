"""Live remote-PPG heart rate from a face video.

Face-visible video -> SegFace face parsing -> rPPG on skin -> band-pass -> live HR.
The top of the frame shows four panels: (left) the full SegFace face parsing (all 19
CelebAMask-HQ classes, colorized) and the extracted skin ROI; (right) the live rPPG
pulse signal and, below it, the HR panel (waveform + bpm + SNR).

A single SegFace pass (``FaceSkinExtractor.analyze``) gives both the full face
parsing and the skin ROI. The pulse is sampled from the model-segmented facial
skin (SegFace's ``skin`` class -- excludes eyes, brows, nose, lips, mouth, hair,
neck). No hard-coded ROIs and no separate face detector (``Segmentation.Face``
detects faces itself). Standalone on purpose: the full-inference pipeline targets
footage where the face is not clearly visible.

Usage:
    python rppg_heartrate.py face_video.mp4 --method POS --show
    python rppg_heartrate.py face_video.mp4 --seg_every 5 --hr_band 0.75 4.0 --output_dir output
"""

import argparse
from pathlib import Path

import cv2
from physiotrack.signals import (
    HeartRateEstimator, HeartRatePlotter, RPPGPlotter, FaceSkinExtractor,
)
from physiotrack.capture.orientation import resolve_rotation, apply_rotation


def _panel(canvas, title, width):
    """Down-scale an image-res canvas to a labelled side panel (font scales with width)."""
    h, w = canvas.shape[:2]
    panel = cv2.resize(canvas, (width, max(1, int(h * width / w))))
    s = width / 300.0
    cv2.putText(panel, title, (int(8 * s), int(20 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * s, (235, 235, 235), max(1, round(s)), cv2.LINE_AA)
    return panel


def run(video_path, method="POS", hr_band=(0.75, 4.0), window_sec=10.0,
        seg_every=3, rotate=0, output_dir=None, show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rot = resolve_rotation(rotate)                  # explicit angle only; 0 = no rotation
    if rot:
        print(f"[orientation] rotating frames {rot} deg (--rotate).")

    # Overlay sizes are fractions of the (post-rotation) video width, so they look the
    # same on any resolution. ~60% larger than the original fixed panels.
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = rh if rot in (90, 270) else rw
    margin = max(8, int(0.012 * W))
    panel_w = max(120, int(0.20 * W))               # each face panel
    hr_w = max(240, int(0.36 * W)); hr_h = int(hr_w * 170 / 460)

    extractor = FaceSkinExtractor(device=0)        # SegFace skin (detects faces itself)
    # One rPPG core; both panels read from it (HR is derived from the same BVP signal).
    est = HeartRateEstimator(method, fps, hr_band=tuple(hr_band), window_sec=window_sec)
    rppg_plot = RPPGPlotter(estimator=est, canvas_width=hr_w, canvas_height=hr_h)
    hr_plot = HeartRatePlotter(estimator=est, canvas_width=hr_w, canvas_height=hr_h)

    # Writer is created lazily on the first (rotated) frame, since 90/270 swaps W/H.
    writer, out_path = None, None
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{Path(video_path).stem}_hr.mp4"

    skin_mask, skin_canvas, parsing_canvas, fidx = None, None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = apply_rotation(frame, rot)
        if out_path is not None and writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        # Re-segment every `seg_every` frames (face region changes slowly); reuse between.
        # One SegFace pass yields the skin ROI (for rPPG) and the full face parsing.
        if fidx % max(1, seg_every) == 0:
            fp = extractor.analyze(frame)
            skin_mask, skin_canvas, parsing_canvas = fp.skin_mask, fp.skin_canvas, fp.parsing_canvas

        if skin_mask is not None and skin_mask.any():
            est.update(frame, roi_mask=skin_mask, frame_time=fidx / fps)        # rPPG on segmented skin

        # top-left (left -> right): full face parsing, then skin ROI.
        x = margin
        for canvas, title in ((parsing_canvas, "Face parsing (SegFace)"),
                              (skin_canvas, "Skin ROI (SegFace)")):
            if canvas is None:
                continue
            panel = _panel(canvas, title, panel_w)
            ph, pw = panel.shape[:2]
            if x + pw <= frame.shape[1] and ph + margin <= frame.shape[0]:
                frame[margin:margin + ph, x:x + pw] = panel
                x += pw + margin

        # top-right, stacked: the rPPG pulse signal (top) + the HR panel (below).
        frame = rppg_plot.attach_to_frame(frame, position="top_right", margin=margin)
        frame = hr_plot.attach_to_frame(frame, position="top_right", margin=margin,
                                        above_element_height=rppg_plot.canvas_height)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("rPPG heart rate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done. Final HR estimate: {est.hr} bpm (method={method})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live rPPG heart rate from a face video (SegFace skin ROI)")
    ap.add_argument("video_path")
    ap.add_argument("--method", default="POS", choices=["POS", "CHROM", "LGI", "OMIT"])
    ap.add_argument("--hr_band", type=float, nargs=2, default=[0.75, 4.0],
                    metavar=("LO_HZ", "HI_HZ"), help="HR analysis / band-pass band (Hz)")
    ap.add_argument("--window_sec", type=float, default=10.0, help="sliding window length (s)")
    ap.add_argument("--seg_every", type=int, default=3, help="re-run SegFace every N frames")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="orientation fix in degrees (default 0 = none); pass an explicit "
                         "angle when the clip needs it, e.g. --rotate 180")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.video_path, a.method, a.hr_band, a.window_sec, a.seg_every, a.rotate, a.output_dir, a.show)
