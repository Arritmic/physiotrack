# Installation

Physiotrack is a pure-Python package installed from source. It builds on PyTorch
and OpenCV, and pulls in the neural backends (Ultralytics, ViTPose, Sapiens,
Depth-Anything-V2, MotionBERT, SegFace, 6DRepNet360) through its dependencies.

!!! info "Requirements at a glance"
    | Requirement | Recommendation |
    |-------------|----------------|
    | **Python**  | 3.8+ (tested on 3.11) |
    | **PyTorch** | 2.x — install the build matching your platform/CUDA *first* |
    | **OS**      | Linux, Windows, macOS |
    | **Hardware**| CPU works; a CUDA GPU is recommended for real-time/video |
    | **Weights** | Auto-downloaded from Hugging Face / Ultralytics on first use, cached outside the package |

## Install PyTorch first

Physiotrack does not pin a PyTorch build for you — install the wheel that matches
your platform and CUDA version before installing the package. The repo's
`requirements.txt` pins a **CUDA 12.8** build (`torch==2.9.1`,
`torchvision==0.24.1`); adjust the index URL for a different CUDA version or for
CPU-only.

=== "CUDA GPU"
    ```bash
    pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu128
    ```

=== "CPU only"
    ```bash
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    ```

!!! tip "Which device string?"
    Pass `device="cpu"` to run on CPU, or `device=0` / `device="cuda"` for the
    first CUDA GPU. `device="mps"` targets Apple Silicon. Every predictor and the
    `Video` orchestrator accept a `device` argument. See
    [Core Concepts](concepts.md#devices).

## Install Physiotrack

=== "From source (editable)"
    ```bash
    git clone https://github.com/tharindu326/physiotrack.git
    cd physiotrack
    pip install -e .
    ```

=== "Directly from GitHub"
    ```bash
    pip install "git+https://github.com/tharindu326/physiotrack.git"
    ```

An editable install (`-e`) is convenient if you plan to read or modify the source
and follow along with the scripts in [`examples/`](https://github.com/tharindu326/physiotrack/tree/main/examples).

## Core dependencies

`pip install -e .` resolves the runtime stack. The main pieces are:

| Package | Role |
|---------|------|
| `torch`, `torchvision`, `xformers` | Deep-learning backends (ViTPose, Sapiens, MotionBERT, SegFace, Depth-Anything-V2) |
| `ultralytics` | YOLO11 / RT-DETR detection, pose and segmentation |
| `opencv-python` | Frame I/O, drawing, video capture |
| `numpy`, `pandas`, `einops`, `timm`, `easydict` | Array math, model plumbing |
| `gdown`, `imageio`, `pillow` | Weight download & image handling |
| `lap`, `cython_bbox`, `fastdtw` | Tracking association & signal metrics |
| `openpyxl` | Spreadsheet export for evaluation scripts |

!!! note "Model weights are not bundled"
    Nothing large ships in the package. The first time you construct a predictor,
    its weights **auto-download** through the [`Models`][physiotrack.Models]
    registry — Ultralytics weights via `ultralytics`, everything else from Hugging
    Face — and are cached locally for subsequent runs. This means the very first
    call to a new model needs a network connection.

## Optional: video codec (OpenH264)

The [`Video`][physiotrack.Video] orchestrator writes annotated MP4s using the
H.264 (`avc1`) codec. On some platforms (notably Windows) OpenCV ships without the
OpenH264 runtime, so encoding fails. The repo includes a helper that downloads and
installs the codec next to your OpenCV install:

```bash
python install_openh264.py
```

??? note "What the helper does"
    It fetches `openh264-1.8.0-win64.dll` from Cisco's release page, decompresses
    it, and drops it into your `cv2` package directory so OpenCV can find the
    `avc1` encoder. Only needed if you see H.264 codec warnings when writing video;
    image-only workflows do not require it.

## Verify the install

```python
import physiotrack as pt

print(pt.__all__)                      # public API surface
det = pt.Detection.Person(device="cpu")
print(det)                             # first run downloads YOLO person weights
```

If the import succeeds and the detector constructs (downloading its weights on the
first run), you are ready for the [Quickstart](quickstart.md).
