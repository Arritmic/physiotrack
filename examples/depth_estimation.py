"""
Depth Estimation Example
Demonstrates how to use DepthAnythingV2 for monocular depth estimation.
"""

from physiotrack import Depth, Models
import cv2
import argparse
import numpy as np


def run_depth_estimation(image_path: str, device: int = 0, save_output: bool = True):
    """
    Run depth estimation on an image.

    Args:
        image_path: Path to input image
        device: GPU device ID
        save_output: Whether to save the output images
    """
    print("="*60)
    print("Depth Estimation with DepthAnythingV2")
    print("="*60)

    print(f"Model: DepthAnythingV2 ViT-B")
    print(f"Device: {'cuda:' + str(device) if device >= 0 else 'cpu'}")

    # Initialize depth estimator
    depth_estimator = Depth.Custom(
        model=Models.Depth.DepthAnythingV2.vitb,
        device=device,
        input_size=518,
        verbose=True
    )

    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Run depth estimation
    print("\nRunning depth estimation...")

    # Get raw depth and colored depth map
    raw_depth, colored_depth = depth_estimator.estimate(image, normalize=True, colormap='inferno')

    print(f"Inference time: {depth_estimator.get_avg_inference_time():.2f}ms")
    print(f"FPS: {depth_estimator.get_avg_fps():.2f}")

    # Save outputs
    if save_output:
        import os
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path) or '.'

        # Save raw depth (normalized to 0-255)
        raw_depth_path = os.path.join(output_dir, f"{base_name}_depth_raw.png")
        cv2.imwrite(raw_depth_path, raw_depth)
        print(f"\nSaved raw depth map: {raw_depth_path}")

        # Save colored depth
        colored_depth_path = os.path.join(output_dir, f"{base_name}_depth_colored.png")
        cv2.imwrite(colored_depth_path, colored_depth)
        print(f"Saved colored depth map: {colored_depth_path}")

        # Create side-by-side comparison
        h, w = image.shape[:2]
        colored_depth_resized = cv2.resize(colored_depth, (w, h))
        comparison = np.hstack([image, colored_depth_resized])
        comparison_path = os.path.join(output_dir, f"{base_name}_depth_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"Saved comparison: {comparison_path}")

    print("\n" + "="*60)
    print("Depth estimation complete!")
    print("="*60)

    return raw_depth, colored_depth


def run_video_depth_estimation(video_path: str, device: int = 0, show: bool = True):
    """
    Run depth estimation on a video.

    Args:
        video_path: Path to input video
        device: GPU device ID
        show: Whether to display the output
    """
    print("="*60)
    print("Video Depth Estimation with DepthAnythingV2")
    print("="*60)

    # Initialize depth estimator
    depth_estimator = Depth.Custom(
        model=Models.Depth.DepthAnythingV2.vitb,
        device=device,
        input_size=518,
        verbose=True
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video path
    import os
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(video_path) or '.'
    output_path = os.path.join(output_dir, f"{base_name}_depth.mp4")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"\nProcessing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run depth estimation
        raw_depth, colored_depth = depth_estimator.estimate(frame, normalize=True, colormap='inferno')

        # Resize to original size
        colored_depth_resized = cv2.resize(colored_depth, (w, h))

        # Write frame to output video
        out.write(colored_depth_resized)

        # Print progress
        if frame_count % 30 == 0:
            print(f"Processing: {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

        if show:
            cv2.imshow('Depth Estimation', colored_depth_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nVideo processing complete!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Depth estimation with DepthAnythingV2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image depth estimation
  python depth_estimation.py image.jpg

  # Video depth estimation
  python depth_estimation.py video.mp4 --video
        """
    )
    parser.add_argument('input', type=str, help='Path to input image or video')
    parser.add_argument('--video', action='store_true', help='Process as video')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--no-save', action='store_true', help='Do not save output images')
    parser.add_argument('--no-show', action='store_true', help='Do not display output (video only)')

    args = parser.parse_args()

    if args.video:
        run_video_depth_estimation(args.input, args.device, not args.no_show)
    else:
        run_depth_estimation(args.input, args.device, not args.no_save)
