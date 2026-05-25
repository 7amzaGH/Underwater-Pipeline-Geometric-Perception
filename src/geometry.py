"""
geometry.py
-----------
PCA-based geometric parameter extraction for underwater pipeline masks.

This module converts a binary segmentation mask into navigation-relevant
image-plane geometric cues:

- pipeline center position
- lateral offset from image center
- dominant orientation angle
- discrete direction command
"""

import cv2
import numpy as np


def estimate_pipeline_geometry(
    mask,
    angle_threshold=5.0,
    min_pixels=50
):
    """
    Estimate pipeline geometry from a binary segmentation mask.

    Args:
        mask (np.ndarray): Binary mask where pipeline pixels are 1 or 255.
        angle_threshold (float): Angle threshold in degrees for LEFT/RIGHT decision.
        min_pixels (int): Minimum number of foreground pixels required for valid PCA.

    Returns:
        dict: Geometry output containing:
            valid: bool
            center: (x, y)
            center_x: float
            center_y: float
            offset_x: float
            angle_deg: float
            direction: str
            axis: (dx, dy) or None
    """

    if mask is None:
        return _invalid_geometry()

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)

    h, w = binary.shape[:2]
    ys, xs = np.where(binary > 0)

    if len(xs) < min_pixels:
        return _invalid_geometry(width=w, height=h)

    points = np.column_stack((xs, ys)).astype(np.float32)

    center = points.mean(axis=0)
    center_x, center_y = center

    centered = points - center
    cov = np.cov(centered.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    dx, dy = principal_axis

    # Normalize arrow direction visually upward.
    if dy > 0:
        dx, dy = -dx, -dy

    angle_deg = float(np.degrees(np.arctan2(dx, -dy)))
    offset_x = float(center_x - (w / 2.0))

    if angle_deg > angle_threshold:
        direction = "RIGHT"
    elif angle_deg < -angle_threshold:
        direction = "LEFT"
    else:
        direction = "STRAIGHT"

    return {
        "valid": True,
        "center": (float(center_x), float(center_y)),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "offset_x": offset_x,
        "angle_deg": angle_deg,
        "direction": direction,
        "axis": (float(dx), float(dy)),
        "num_pixels": int(len(xs)),
    }


def _invalid_geometry(width=640, height=640):
    """
    Return safe fallback geometry when mask is invalid.
    """

    return {
        "valid": False,
        "center": (width / 2.0, height / 2.0),
        "center_x": width / 2.0,
        "center_y": height / 2.0,
        "offset_x": 0.0,
        "angle_deg": 0.0,
        "direction": "STRAIGHT",
        "axis": None,
        "num_pixels": 0,
    }


def compute_geometry_error(pred_geometry, gt_geometry):
    """
    Compute center and angle error between predicted and ground-truth geometry.

    Args:
        pred_geometry (dict): Output of estimate_pipeline_geometry().
        gt_geometry (dict): Output of estimate_pipeline_geometry().

    Returns:
        dict: center_error_px, angle_error_deg, direction_match.
    """

    if not pred_geometry["valid"] or not gt_geometry["valid"]:
        return {
            "valid": False,
            "center_error_px": None,
            "angle_error_deg": None,
            "direction_match": False,
        }

    pred_center = np.array(pred_geometry["center"])
    gt_center = np.array(gt_geometry["center"])

    center_error = float(np.linalg.norm(pred_center - gt_center))

    angle_error = abs(pred_geometry["angle_deg"] - gt_geometry["angle_deg"])
    angle_error = min(angle_error, 180.0 - angle_error)

    direction_match = pred_geometry["direction"] == gt_geometry["direction"]

    return {
        "valid": True,
        "center_error_px": center_error,
        "angle_error_deg": float(angle_error),
        "direction_match": direction_match,
    }


def draw_geometry(
    frame,
    geometry,
    arrow_color=(0, 0, 255),
    center_color=(0, 255, 255),
    text_color=(255, 255, 255),
    line_color=(180, 180, 180),
    arrow_scale=0.28
):
    """
    Draw center point, PCA arrow, centerline, offset line, and text values.

    Args:
        frame (np.ndarray): BGR frame.
        geometry (dict): Output from estimate_pipeline_geometry().
        arrow_color (tuple): BGR arrow color.
        center_color (tuple): BGR center/offset color.
        text_color (tuple): BGR text color.
        line_color (tuple): BGR image-center line color.
        arrow_scale (float): Arrow length relative to image height.

    Returns:
        np.ndarray: Annotated frame.
    """

    vis = frame.copy()
    h, w = vis.shape[:2]

    cx = int(geometry["center_x"])
    cy = int(geometry["center_y"])

    cv2.line(vis, (w // 2, 0), (w // 2, h), line_color, 1, cv2.LINE_AA)

    cv2.circle(vis, (cx, cy), 8, center_color, -1)
    cv2.circle(vis, (w // 2, cy), 6, (255, 255, 255), -1)

    cv2.line(
        vis,
        (w // 2, cy),
        (cx, cy),
        center_color,
        3,
        cv2.LINE_AA
    )

    if geometry["axis"] is not None:
        dx, dy = geometry["axis"]
        arrow_len = int(arrow_scale * h)

        end_x = int(cx + dx * arrow_len)
        end_y = int(cy + dy * arrow_len)

        cv2.arrowedLine(
            vis,
            (cx, cy),
            (end_x, end_y),
            arrow_color,
            4,
            tipLength=0.25,
            line_type=cv2.LINE_AA
        )

    direction = geometry["direction"]

    if direction == "LEFT":
        direction_color = (255, 180, 0)
    elif direction == "RIGHT":
        direction_color = (0, 180, 255)
    else:
        direction_color = (0, 255, 0)

    cv2.rectangle(vis, (15, 15), (430, 165), (0, 0, 0), -1)
    cv2.rectangle(vis, (15, 15), (430, 165), (80, 80, 80), 2)

    cv2.putText(
        vis,
        "Pipeline Geometry",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        text_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"Center x: {geometry['center_x']:.1f} px",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"Offset: {geometry['offset_x']:.1f} px",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"Angle: {geometry['angle_deg']:.2f} deg",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA
    )

    cv2.rectangle(vis, (455, 15), (685, 165), direction_color, -1)

    cv2.putText(
        vis,
        direction,
        (480, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    return vis