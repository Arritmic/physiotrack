"""Contactless vitals from a face video: rPPG signal, heart rate, HRV and respiration.

Face-visible video -> SegFace face parsing -> rPPG on the segmented skin -> band-pass ->
the full pulse-analysis stack, all overlaid on one frame:

* top-left  : the full SegFace face parsing (19 CelebAMask-HQ classes, colorized) and
              the extracted skin ROI.
* top-right : the live rPPG pulse (BVP), the heart rate (bpm + SNR), the heart-rate
              variability grid (RMSSD/SDNN/pNN50/SD1/SD2/LF-HF) and the respiration rate
              (breaths/min).

A single SegFace pass (``FaceSkinExtractor.analyze``) gives both the face parsing and
the skin ROI. A single [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator]
drives every panel via the shared-estimator pattern, so the rPPG is computed exactly
once per frame: the estimator is updated once, and the HRV / respiration panels only
``refresh()`` their cached values. HRV needs a longer window than HR, so a 60 s window
is used by default. The pulse is sampled from the model-segmented facial skin (SegFace's
``skin`` class -- excludes eyes, brows, nose, lips, mouth, hair, neck); no hard-coded
ROIs and no separate face detector (``Segmentation.Face`` detects faces itself).

Standalone on purpose: the full-inference pipeline (``full_inference.py --rppg --hrv
--respiration``) targets footage where the face is not clearly visible.

Usage:
    python rppg_vitals.py face_video.mp4 --show
    python rppg_vitals.py face_video.mp4 --method POS --window_sec 60 --output_dir output
"""

import argparse
from pathlib import Path

import cv2
from physiotrack.signals import (
    HeartRateEstimator, HeartRatePlotter, RPPGPlotter, HRVPlotter,
    RespirationPlotter, FaceSkinExtractor,
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


def run(video_path, method="POS", hr_band=(0.75, 4.0), window_sec=60.0,
        seg_every=3, rotate=0, output_dir=None, show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rot = resolve_rotation(rotate)                  # explicit angle only; 0 = no rotation
    if rot:
        print(f"[orientation] rotating frames {rot} deg (--rotate).")

    # Overlay sizes are fractions of the (post-rotation) video width, so they look the
    # same on any resolution.
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = rh if rot in (90, 270) else rw
    margin = max(8, int(0.012 * W))
    panel_w = max(120, int(0.20 * W))               # each face panel (top-left)
    pw = max(240, int(0.36 * W)); ph = int(pw * 170 / 460)   # right-side vitals panels

    extractor = FaceSkinExtractor(device=0)          # SegFace skin (detects faces itself)
    # One rPPG core; every panel reads from it, so the pulse is computed once per frame.
    est = HeartRateEstimator(method, fps, hr_band=tuple(hr_band), window_sec=window_sec)
    rppg_plot = RPPGPlotter(estimator=est, canvas_width=pw, canvas_height=ph)
    hr_plot = HeartRatePlotter(estimator=est, canvas_width=pw, canvas_height=ph)
    hrv_plot = HRVPlotter(estimator=est, canvas_width=pw, canvas_height=int(pw * 190 / 460))
    resp_plot = RespirationPlotter(estimator=est, canvas_width=pw, canvas_height=ph)

    # Writer is created lazily on the first (rotated) frame, since 90/270 swaps W/H.
    writer, out_path = None, None
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{Path(video_path).stem}_vitals.mp4"

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

        # Update the shared estimator ONCE, then refresh the derived (HRV/resp) panels.
        if skin_mask is not None and skin_mask.any():
            est.update(frame, roi_mask=skin_mask, frame_time=fidx / fps)   # rPPG on segmented skin
        hrv_plot.refresh()
        resp_plot.refresh()

        # top-left (left -> right): full face parsing, then skin ROI.
        x = margin
        for canvas, title in ((parsing_canvas, "Face parsing (SegFace)"),
                              (skin_canvas, "Skin ROI (SegFace)")):
            if canvas is None:
                continue
            panel = _panel(canvas, title, panel_w)
            ph2, pw2 = panel.shape[:2]
            if x + pw2 <= frame.shape[1] and ph2 + margin <= frame.shape[0]:
                frame[margin:margin + ph2, x:x + pw2] = panel
                x += pw2 + margin

        # top-right, stacked: rPPG pulse -> HR -> HRV grid -> respiration.
        y = 0
        for panel in (rppg_plot, hr_plot, hrv_plot, resp_plot):
            frame = panel.attach_to_frame(frame, position="top_right", margin=margin,
                                          above_element_height=y)
            y += panel.canvas_height + margin

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("Contactless vitals", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    hrv = est.hrv()
    print(f"Done (method={method}).")
    print(f"  Final HR    : {est.hr} bpm   (SNR {est.snr} dB)")
    print(f"  Respiration : {est.respiration_rate('pulse'):.1f} breaths/min")
    if hrv:
        print(f"  HRV  RMSSD={hrv.get('RMSSD'):.1f} ms  SDNN={hrv.get('SDNN'):.1f} ms  "
              f"SD1={hrv.get('SD1'):.1f} ms  LF/HF={hrv.get('LFHF'):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Contactless vitals from a face video: rPPG pulse, HR, HRV and respiration")
    ap.add_argument("video_path")
    ap.add_argument("--method", default="POS", choices=["POS", "CHROM", "LGI", "OMIT"])
    ap.add_argument("--hr_band", type=float, nargs=2, default=[0.75, 4.0],
                    metavar=("LO_HZ", "HI_HZ"), help="HR analysis / band-pass band (Hz)")
    ap.add_argument("--window_sec", type=float, default=60.0,
                    help="sliding window length (s); HRV needs a longer window than HR")
    ap.add_argument("--seg_every", type=int, default=3, help="re-run SegFace every N frames")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="orientation fix in degrees (default 0 = none); pass an explicit "
                         "angle when the clip needs it, e.g. --rotate 180")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run(a.video_path, a.method, a.hr_band, a.window_sec, a.seg_every, a.rotate, a.output_dir, a.show)
