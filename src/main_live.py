"""
main_live.py
------------
Live underwater pipeline perception launcher.

Designed as a proposed deployment entry point for camera-based
pipeline navigation perception.

Usage:
    python src/main_live.py --model models/best.onnx --backend onnx --camera 0
"""

import argparse
import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detect import PTDetector, ONNXDetector, draw_detections, get_best_detection
from geometry import estimate_pipeline_geometry, draw_geometry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live underwater pipeline perception launcher"
    )

    parser.add_argument("--model", required=True, help="Model path: .pt or .onnx")
    parser.add_argument("--backend", default="onnx", choices=["pt", "onnx"])
    parser.add_argument("--provider", default="CPUExecutionProvider")

    parser.add_argument("--camera", default="0", help="Camera index or video stream URL")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--angle-threshold", type=float, default=5.0)
    parser.add_argument("--min-pixels", type=int, default=50)

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", default="outputs/live_demo.mp4")

    return parser.parse_args()


def parse_camera_source(camera_arg):
    try:
        return int(camera_arg)
    except ValueError:
        return camera_arg


def make_detector(args):
    if args.backend == "pt":
        return PTDetector(args.model, conf_threshold=args.conf)

    return ONNXDetector(
        args.model,
        provider=args.provider,
        conf_threshold=args.conf
    )


def main():
    args = parse_args()

    detector = make_detector(args)

    camera_source = parse_camera_source(args.camera)
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/source: {args.camera}")

    writer = None

    if args.save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    print("[INFO] Live pipeline perception started.")
    print("[INFO] Press 'q' to quit.")

    frame_id = 0

    try:
        while True:
            start = time.time()

            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Empty frame received.")
                break

            detections = detector.detect(frame)
            best = get_best_detection(detections)

            if best is not None:
                geometry = estimate_pipeline_geometry(
                    best["mask"],
                    angle_threshold=args.angle_threshold,
                    min_pixels=args.min_pixels
                )

                vis = draw_detections(frame, [best])
                vis = draw_geometry(vis, geometry)

                print(
                    f"Frame {frame_id:06d} | "
                    f"conf={best['conf']:.2f} | "
                    f"center_x={geometry['center_x']:.1f} | "
                    f"offset={geometry['offset_x']:.1f}px | "
                    f"angle={geometry['angle_deg']:.2f}deg | "
                    f"direction={geometry['direction']}"
                )

            else:
                vis = frame.copy()

                cv2.putText(
                    vis,
                    "No pipeline detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )

                print(f"Frame {frame_id:06d} | no pipeline detected")

            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (30, vis.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if writer is not None:
                writer.write(vis)

            cv2.imshow("Live Pipeline Perception", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()

        if writer is not None:
            writer.release()
            print(f"[DONE] Saved live output: {args.output}")

        cv2.destroyAllWindows()
        print("[INFO] Live pipeline perception stopped.")


if __name__ == "__main__":
    main()