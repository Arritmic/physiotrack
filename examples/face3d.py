"""
Example script demonstrating face orientation estimation with visualization.
"""

import cv2
from physiotrack import VRFace, FaceOrientation, Models
from physiotrack.face import draw_axis, plot_pose_cube

# Load image
image_path = 'kinect_s1_v1_frame1.png'  # Change this to your image path
img = cv2.imread(image_path)

# Initialize face detector and face orientation estimator
face_detector = VRFace(device=0)
face_orientation = FaceOrientation(model=Models.Pose3D.FaceOrientation.VR, device=0)

# Detect faces
det_result = face_detector.predict(img)   # or face_detector(img)
bboxes = det_result.boxes                  # (N, 4) ndarray

# Estimate face orientation
result = face_orientation.predict(img, bboxes)

# Simplest visualization: let the result draw the orientation axes
annotated = result.plot()
cv2.imwrite('face_orientation_output.png', annotated)

# Manual visualization (equivalent), iterating over instances
vis_img = img.copy()
for inst in result:
    pose = inst.orientation   # {"yaw": .., "pitch": .., "roll": ..}
    bbox = inst.box

    x1, y1, x2, y2 = bbox
    face_center_x = int((x1 + x2) / 2)
    face_center_y = int((y1 + y2) / 2)

    # Draw bounding box
    # cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    face_width = x2 - x1
    face_height = y2 - y1

    # Draw direction axes
    axis_size = max(face_width, face_height) * 0.6  # Scale axes to 60% of larger face dimension
    vis_img = draw_axis(
        vis_img,
        yaw=pose['yaw'],
        pitch=pose['pitch'],
        roll=pose['roll'],
        tdx=face_center_x,
        tdy=face_center_y,
        size=axis_size
    )

    # Draw pose cube (alternative visualization)
    # cube_size = max(face_width, face_height) * 0.6  # Scale cube to 60% of larger face dimension
    # vis_img = plot_pose_cube(
    #     vis_img,
    #     yaw=pose['yaw'],
    #     pitch=pose['pitch'],
    #     roll=pose['roll'],
    #     tdx=face_center_x,
    #     tdy=face_center_y,
    #     size=cube_size
    # )

# Save output
cv2.imwrite('face_orientation_output_manual.png', vis_img)
