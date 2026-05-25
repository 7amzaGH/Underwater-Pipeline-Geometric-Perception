"""
detect.py
---------
YOLOv8n-seg inference wrapper for underwater pipeline segmentation.

Supports:
- PyTorch / Ultralytics .pt inference
- ONNX Runtime inference for deployment-style evaluation

Output format:
[
    {
        "bbox": (x1, y1, x2, y2),
        "conf": 0.95,
        "mask": binary_mask,   # uint8 mask, same size as original frame
    }
]
"""

import cv2
import numpy as np


class PTDetector:
    """
    PyTorch / Ultralytics YOLOv8 segmentation inference.

    Recommended for:
    - training validation
    - qualitative analysis
    - FP32 reference inference
    """

    def __init__(self, model_path, conf_threshold=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install 'ultralytics' to use PTDetector.")

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Run YOLOv8 segmentation inference.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of detections with bbox, confidence, and binary mask.
        """

        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None or results.masks is None:
            return detections

        boxes = results.boxes
        masks = results.masks.data.cpu().numpy()

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            mask = masks[i]
            mask = cv2.resize(mask, (w, h))
            mask = (mask > 0.5).astype(np.uint8)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "conf": round(conf, 4),
                "mask": mask
            })

        return detections


class ONNXDetector:
    """
    ONNX Runtime YOLOv8-seg inference.

    Recommended for:
    - deployment-style inference
    - CPU simulation
    - edge/NPU compatible evaluation

    Important:
    This implementation assumes a standard exported YOLOv8-seg ONNX model
    with two outputs:
        output0: detections
        output1: mask prototypes
    """

    def __init__(
        self,
        model_path,
        provider="CPUExecutionProvider",
        input_size=640,
        conf_threshold=0.5,
        mask_threshold=0.5
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install 'onnxruntime' to use ONNXDetector.")

        available = ort.get_available_providers()
        selected = provider if provider in available else "CPUExecutionProvider"

        self.session = ort.InferenceSession(model_path, providers=[selected])
        self.input_name = self.session.get_inputs()[0].name

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

    def _preprocess(self, frame):
        """
        Resize frame to YOLO input size and normalize.

        Returns:
            blob: model input tensor
            original_shape: original frame size
        """

        h, w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, (h, w)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _crop_mask_to_box(self, mask, box):
        """
        Suppress mask activations outside the detected bounding box.
        This is important for geometry-preserving YOLOv8-seg ONNX decoding.
        """

        x1, y1, x2, y2 = box
        cropped = np.zeros_like(mask, dtype=np.float32)

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(mask.shape[1], int(x2))
        y2 = min(mask.shape[0], int(y2))

        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return cropped

    def detect(self, frame):
        """
        Run ONNX YOLOv8 segmentation inference.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of detections with bbox, confidence, and binary mask.
        """

        blob, (orig_h, orig_w) = self._preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})

        pred = outputs[0]
        proto = outputs[1]

        # YOLOv8-seg ONNX usually outputs:
        # pred:  [1, 37, 8400] for one class + 32 mask coeffs
        # proto: [1, 32, 160, 160]
        pred = np.squeeze(pred)

        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        proto = np.squeeze(proto)

        detections = []

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        for det in pred:
            # YOLOv8-seg one-class layout:
            # [cx, cy, w, h, class_conf, mask_coeff_1...mask_coeff_32]
            cx, cy, bw, bh = det[0:4]
            conf = float(det[4])

            if conf < self.conf_threshold:
                continue

            mask_coeffs = det[5:]

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)

            # Reconstruct prototype mask
            mask = np.matmul(mask_coeffs, proto.reshape(proto.shape[0], -1))
            mask = mask.reshape(proto.shape[1], proto.shape[2])
            mask = self._sigmoid(mask)

            # Resize prototype mask to model input size
            mask = cv2.resize(mask, (self.input_size, self.input_size))

            # Important for YOLOv8-seg deployment decoding
            mask = self._crop_mask_to_box(mask, (x1_i, y1_i, x2_i, y2_i))

            # Resize mask back to original frame size
            mask = cv2.resize(mask, (orig_w, orig_h))
            mask = (mask > self.mask_threshold).astype(np.uint8)

            bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            )

            detections.append({
                "bbox": bbox,
                "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                "conf": round(conf, 4),
                "mask": mask
            })

        return detections


def draw_detections(
    frame,
    detections,
    mask_color=(0, 255, 0),
    box_color=(255, 100, 0),
    alpha=0.45,
    thickness=2
):
    """
    Draw segmentation masks and bounding boxes.

    Args:
        frame: BGR image.
        detections: output from PTDetector or ONNXDetector.
        mask_color: BGR mask overlay color.
        box_color: BGR bounding box color.
        alpha: mask transparency.
        thickness: box thickness.

    Returns:
        Annotated BGR frame.
    """

    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        mask = det["mask"]

        colored_mask = np.zeros_like(annotated)
        colored_mask[mask > 0] = mask_color

        annotated = cv2.addWeighted(
            annotated,
            1.0,
            colored_mask,
            alpha,
            0
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        label = f"pipeline {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
            cv2.LINE_AA
        )

    return annotated


def get_best_detection(detections):
    """
    Return the highest-confidence detection.

    Useful because this project assumes one dominant pipeline instance.
    """

    if not detections:
        return None

    return max(detections, key=lambda d: d["conf"])