"""
Example: SegFace face-part parsing via the unified segmentation API.

`Segmentation.Face` parses faces into 19 CelebAMask-HQ classes
(skin, eyes, brows, nose, lips, hair, ears, glasses, hat, earring, necklace,
neck, cloth, ...). It runs on face crops: if no boxes are passed to `predict`,
faces are auto-detected with a YOLO face detector (like `Pose` auto-detects
people). Pass `boxes=[...]` to parse specific faces.
"""

import cv2
import numpy as np
from physiotrack import Segmentation

# Load image
image_path = 'kinect_s1_v1_frame1.png'   # change to your image path
img = cv2.imread(image_path)

# Initialize the face-parsing segmenter (default = SegFace Swin-Base / CelebA 512)
parser = Segmentation.Face(device=0)     # device=0 for GPU, 'cpu' for CPU

# ---------------------------------------------------------------------------
# Option A: auto-detect faces, then parse each one
# ---------------------------------------------------------------------------
result = parser.predict(img)             # or parser(img)

# result.seg_map is a (H, W) array of face-part class indices (0 = background).
# result.names maps each class index to its label (same as every other Result).
seg_map = result.seg_map
present = [result.names[c] for c in np.unique(seg_map) if c != 0]
print(f"Detected {len(result)} face(s); parts present: {present}")

# result.plot() overlays the parsing using the 19-class palette
cv2.imwrite('face_parsing_output.png', result.plot())

# ---------------------------------------------------------------------------
# Option B: parse a specific face box (skip auto-detection)
# ---------------------------------------------------------------------------
# from physiotrack import VRFace
# boxes = VRFace(device=0).predict(img).boxes      # (N, 4)
# result = parser.predict(img, boxes=boxes)
# cv2.imwrite('face_parsing_output.png', result.plot())

# ---------------------------------------------------------------------------
# Per-face access: each instance carries the face box; slice seg_map to get
# that face's part map.
# ---------------------------------------------------------------------------
for inst in result:
    x1, y1, x2, y2 = inst.box.astype(int)
    face_parts = seg_map[y1:y2, x1:x2]
    print(f"face @ {inst.box.astype(int).tolist()} -> "
          f"{len(np.unique(face_parts)) - 1} part classes")
