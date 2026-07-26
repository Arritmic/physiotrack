"""Lift 2D pose keypoints to 3D, with both integrated backends.

`Pose3D.predict` takes keypoints in memory and returns a `Pose3DResult`, so the 2D
pass feeds straight into the lifter without a JSON round-trip. Use `predict_json`
instead when the 2D pass already wrote a file (see the bottom of this script).
"""
from physiotrack import CanonicalView, Models, Pose, Pose3D, Video

VIDEO = "BV_S17_cut1.mp4"

# ------------------- 2D pass: keypoints stay in memory -------------------

results = Video(source=VIDEO, pose=Pose.Person()).run()
fps = results[0].meta.fps if results else 30

# ------------------- MotionBERT -------------------

pose3D = Pose3D(
    model=Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
    device='cuda',
    clip_len=243,
    pixel=False,
)
poses = pose3D.predict(results, fps=fps, canonical_view=CanonicalView.FRONT)

print(f"MotionBERT: {poses}")
print(f"  shape           : {poses.poses.shape}")
print(f"  left wrist path : {poses.by_name('left_wrist').shape}")

# ------------------- DDHPose -------------------
# DDHPose normalises pixel coordinates, so it needs the source frame size.

pose3D = Pose3D(
    model=Models.Pose3D.DDH.best,
    device='cuda',
    num_proposals=10,
    sampling_timesteps=5,
)
poses = pose3D.predict(
    results,
    fps=fps,
    frame_size=(1920, 1080),
    batch_size=8,
    canonical_view=CanonicalView.FRONT,
)
print(f"DDHPose: {poses}")

# ------------------- Offline: lift an existing 2D-pose JSON -------------------
# `predict_json` additionally writes the rendered .mp4, the .npy array and a
# *_with_3d_keypoints.json when `out_path` is set.
#
# pose3D = Pose3D(model=Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite,
#                 device='cuda', render_video=True, save_npy=True)
# frames_data, poses = pose3D.predict_json(
#     "output/BV_S17_cut1_result.json", VIDEO,
#     out_path="output/", canonical_view=CanonicalView.FRONT,
# )
