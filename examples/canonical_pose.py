"""Four ways to canonicalize 3D poses to a fixed viewpoint."""

from pathlib import Path

from physiotrack import (CanonicalView, Models, Pose, Pose3D, PoseCanonicalizer,
                         Video, canonicalize_pose)

VIDEO = 'BV_S17_cut1.mp4'
OUTPUT_DIR = 'output/'
Canon = Models.Pose3D.Canonicalizer

# 2D pass, then lift -- keypoints stay in memory.
results = Video(source=VIDEO, pose=Pose.Person()).run()
fps = results[0].meta.fps if results else 30

pose3D = Pose3D(
    model=Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
    device='cuda',
    clip_len=243,
)

# =============================================================================
# Method 1: Integrated - canonicalize during lifting
# =============================================================================
canonical = pose3D.predict(
    results,
    fps=fps,
    canonical_view=CanonicalView.FRONT,
    canonical_model=Canon.Models.GEOMETRIC,
)
print(f"Integrated: {canonical}")

# =============================================================================
# Method 2: Direct - canonicalize an in-memory array
# =============================================================================
raw = pose3D.predict(results, fps=fps)          # no canonical_view
canonical_poses = canonicalize_pose(
    raw.poses,
    model=Canon.Models.GEOMETRIC,
    view=CanonicalView.FRONT,
)
print(f"Direct: {canonical_poses.shape}")

# =============================================================================
# Method 3: File-based - process a saved .npy file
# =============================================================================
npy_file = 'output/X3D_20250109_120000.npy'
if Path(npy_file).exists():
    canonical_from_file = PoseCanonicalizer.process_npy_file(
        npy_file,
        output_path='output/X3D_canonical.npy',
        view=CanonicalView.FRONT,
        model=Canon.Models.GEOMETRIC,
    )

# =============================================================================
# Method 4: 3DPCNet - the learned canonicalizer (front view only)
# =============================================================================
for name, model in (("S2", Canon.Models._3DPCNetS2), ("S3", Canon.Models._3DPCNetS3)):
    try:
        canonical_3dpcnet = canonicalize_pose(
            raw.poses, model=model, view=CanonicalView.FRONT
        )
        print(f"3DPCNet {name} canonicalized shape: {canonical_3dpcnet.shape}")
    except Exception as e:
        print(f"3DPCNet {name} not available: {e}")
