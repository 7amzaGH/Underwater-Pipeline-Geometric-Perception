"""
main_offline.py
---------------
Offline underwater pipeline perception demo.

Processes a recorded video and saves:
- annotated output video
- per-frame geometry CSV

Usage:
    python src/main_offline.py --video data/demo.mp4 --model models/best.pt
"""

import argparse
import csv
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detect import PTDetector, ONNXDetector, draw_detections, get_best_detection
from geometry import estimate_pipeline_geometry, draw_geometry


CSV_FIELDS = [
    "frame_id",
    "valid",
    "center_x",
    "center_y",
    "offset_x",
    "angle_deg",
    "direction",
    "confidence",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline underwater pipeline perception"
    )

    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="Model path: .pt or .onnx")
    parser.add_argument("--output-video", default="outputs/offline_demo.mp4")
    parser.add_argument("--output-csv", default="outputs/offline_geometry.csv")

    parser.add_argument("--backend", default="pt", choices=["pt", "onnx"])
    parser.add_argument("--provider", default="CPUExecutionProvider")

    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--angle-threshold", type=float, default=5.0)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--display", action="store_true")

    return parser.parse_args()


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

    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    detector = make_detector(args)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    rows = []
    frame_id = 0

    print("[INFO] Starting offline pipeline perception...")

    while True:
        ret, frame = cap.read()

        if not ret:
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

            conf = best["conf"]

        else:
            geometry = estimate_pipeline_geometry(None)
            vis = frame.copy()
            conf = 0.0

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

        rows.append({
            "frame_id": frame_id,
            "valid": geometry["valid"],
            "center_x": round(geometry["center_x"], 3),
            "center_y": round(geometry["center_y"], 3),
            "offset_x": round(geometry["offset_x"], 3),
            "angle_deg": round(geometry["angle_deg"], 3),
            "direction": geometry["direction"],
            "confidence": conf,
        })

        writer.write(vis)

        if args.display:
            cv2.imshow("Pipeline Perception", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(args.output_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"[DONE] Processed frames: {frame_id}")
    print(f"[DONE] Saved video: {args.output_video}")
    print(f"[DONE] Saved CSV:   {args.output_csv}")


if __name__ == "__main__":
    main()