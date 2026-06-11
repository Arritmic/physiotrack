from physiotrack import Pose, Models
import cv2

image = cv2.imread('frame_1.png')
pose = Pose.Person()
result = pose.predict(image)   # auto-detects people; or pose.predict(image, boxes)

print(f"Number of poses detected: {len(result)}")
print(f"Pose architecture: {result.architecture}")

print("\n=== Pose Objects ===")
for i, p in enumerate(result):
    print(f"\nPose {i}:")
    print(f"  ID: {p.id}")
    print(f"  Bounding Box: {p.box}")

    # Print some key keypoints
    print("  Key Keypoints:")
    key_points = ["nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"]
    for kp_name in key_points:
        kp = p.keypoints.by_name(kp_name)
        if kp:
            print(f"    {kp_name}: x={kp.x:.1f}, y={kp.y:.1f}, confidence={kp.confidence:.3f}")

    print("  All visible keypoints (confidence > 0.5):")
    for kp_id in range(133):
        kp = p.keypoints.by_id(kp_id)
        if kp and kp.confidence > 0.5:
            print(f"    {kp.name} (ID:{kp.id}): x={kp.x:.1f}, y={kp.y:.1f}, conf={kp.confidence:.3f}")

annotated = result.plot()
cv2.imwrite('out.png', annotated)
