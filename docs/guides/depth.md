# Depth

Monocular depth estimation predicts a per-pixel depth map from a single RGB frame —
no stereo rig or depth sensor required. Physiotrack wraps
[Depth-Anything-V2](../model-zoo.md), giving you a dense `(H, W)` map you can
normalize, colorize, or feed into 3D reasoning. Use it to add a relative Z channel
to 2D detections, build depth overlays, or gate foreground/background.

Presets live on [`Depth`][physiotrack.Depth]. Call [`predict`][physiotrack.Depth]
(or the instance directly) to get a [`DepthResult`][physiotrack.DepthResult].

## Quick start

```python
from physiotrack import Depth
import cv2

frame = cv2.imread("frame.png")

depth = Depth.DepthAnythingV2Base(device=0)     # 'cpu' also works
result = depth.predict(frame)                    # or: depth(frame)

raw = result.depth                # (H, W) float depth map
norm = result.normalized()        # (H, W) float in [0, 1]
colored = result.plot(colormap="inferno")        # displayable BGR image
cv2.imwrite("depth.png", colored)
```

## Available presets

All presets use the Depth-Anything-V2 backend; they differ only in the ViT
backbone size. See the [Model Zoo](../model-zoo.md) for download details.

| Preset | Backbone | Speed / accuracy |
| --- | --- | --- |
| [`Depth.DepthAnythingV2Small`][physiotrack.Depth.DepthAnythingV2Small] | `vits` | Fastest, least accurate. |
| [`Depth.DepthAnythingV2Base`][physiotrack.Depth.DepthAnythingV2Base] | `vitb` | Balanced. |
| [`Depth.DepthAnythingV2Large`][physiotrack.Depth.DepthAnythingV2Large] | `vitl` | Most accurate. |
| [`Depth.DepthAnythingV2`][physiotrack.Depth.DepthAnythingV2] | `vitl` | Alias for the Large model. |
| [`Depth.Custom`][physiotrack.Depth.Custom] | any | Run any validated `Models.Depth.*`. |

```python
from physiotrack import Depth, Models

# Custom preset: choose an explicit validated model
depth = Depth.Custom(model=Models.Depth.DepthAnythingV2.vitl, device=0)
```

!!! info "Auto-download"
    Weights are pulled from Hugging Face on first use and cached. Larger backbones
    (`vitl`) are slower and heavier — start with `vitb` and scale up if you need it.

## Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `device` | `'cpu'` | `'cpu'`, `'cuda'`, `'mps'`, or an index like `0`. |
| `input_size` | `518` | Square resolution used for inference. |
| `verbose` | `True` | Print initialization info. |

See [`Depth`][physiotrack.Depth] for the full constructor signature.

## Working with results

`predict` returns a [`DepthResult`][physiotrack.DepthResult] for a single frame
(a `list[DepthResult]` for a batch). It exposes the raw map plus two derived views.

```python
result = depth.predict(frame)

result.depth              # (H, W) raw float depth map
result.normalized()       # (H, W) float min-max scaled to [0, 1]
result.plot()             # colorized BGR image (uint8), inferno by default
```

[`DepthResult.plot`][physiotrack.DepthResult.plot] min-max normalizes the map and
applies an OpenCV colormap. Valid `colormap` names are `"inferno"`, `"viridis"`,
`"magma"`, `"plasma"`, and `"jet"`; an unknown name falls back to `"inferno"`.

=== "inferno"

    ```python
    colored = result.plot(colormap="inferno")
    ```

=== "viridis"

    ```python
    colored = result.plot(colormap="viridis")
    ```

=== "jet"

    ```python
    colored = result.plot(colormap="jet")
    ```

Need a grayscale 0–255 map? Scale the normalized view yourself:

```python
import numpy as np
raw_u8 = (result.normalized() * 255).astype(np.uint8)   # (H, W) uint8
```

See [Result objects](../api/results.md) for the full `DepthResult` contract.

## Recipes

!!! example "Video loop"
    Build the estimator once, then colorize each frame.

    ```python
    depth = Depth.DepthAnythingV2Base(device=0)
    cap = cv2.VideoCapture("clip.mp4")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        colored = depth.predict(frame).plot(colormap="inferno")
    ```

!!! tip "Batch inference"
    Pass a list of frames to `predict` to get a `list[DepthResult]`, one per frame.

!!! tip "Side-by-side comparison"
    Resize the colorized depth back to the source size and stack it next to the
    original for a quick visual check.

    ```python
    h, w = frame.shape[:2]
    colored = cv2.resize(result.plot(), (w, h))
    comparison = np.hstack([frame, colored])
    ```

## See also

- [`Depth` API reference](../api/depth.md) — full class and preset docs.
- [Result objects](../api/results.md) — the `DepthResult` contract.
- [Model Zoo](../model-zoo.md) — Depth-Anything-V2 backbones.
- [3D Pose guide](pose3d.md) — combine depth with keypoints for 3D reasoning.
