<div align="center">

# Underwater Pipeline Geometric Perception

### Lightweight Segmentation-Based Underwater Pipeline Perception for Embedded AI Deployment

**Deployment-oriented underwater perception pipeline for extracting navigation-relevant geometric information using YOLOv8n-seg instance segmentation, ONNX inference, and PCA-based geometric estimation.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](#c-embedded-geometry-runtime)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://ultralytics.com)
[![ONNX](https://img.shields.io/badge/ONNXRuntime-Deployment-orange.svg)](https://onnxruntime.ai)
[![INT8](https://img.shields.io/badge/INT8-Quantization-green.svg)](#embedded-deployment)
[![NPU](https://img.shields.io/badge/NPU-Qualcomm%20RB3%20Gen2-blue.svg)](#embedded-deployment)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-purple.svg)](https://universe.roboflow.com/hamzaghitri)
[![Manuscript](https://img.shields.io/badge/Manuscript-Under%20Review-red.svg)](#citation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

★ [IEEE Paper](#) &nbsp;·&nbsp;
★ [Dataset (Roboflow)](https://universe.roboflow.com/hamzaghitri) &nbsp;·&nbsp;
★ [NeptuNet Framework](https://github.com/7amzaGH/NeptuNet-AUV-Intelligent-System)

<br/>
Lightweight underwater perception pipeline operating on monocular underwater imagery for image-plane pipeline center alignment, orientation estimation, and directional cue extraction.
<br/>

<img src="assets/system_demo.gif" alt="Pipeline Demo" width="760"/>

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Embedded Deployment](#embedded-deployment)
- [Datasets](#datasets)
- [Geometry Extraction](#geometry-extraction)
- [Results](#results)
- [Quick Start](#quick-start)
- [Usage in Python](#usage-in-python)
- [Notebook Demo](#notebook-demo)
- [Repository Structure](#repository-structure)
- [Role within NeptuNet](#role-within-neptunet)
- [C++ Embedded Geometry Runtime](#c-embedded-geometry-runtime)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Reliable underwater pipeline perception is a fundamental requirement for subsea robotic inspection missions. Underwater imagery is affected by turbidity, illumination variation, color attenuation, suspended particles, marine biofouling, and low-contrast boundaries.

This repository presents a lightweight deployment-oriented underwater perception pipeline for extracting navigation-relevant geometric information from monocular underwater imagery.

The framework combines:

- **YOLOv8n-seg instance segmentation**
- **deployment-oriented ONNX inference**
- **INT8 embedded deployment**
- **PCA-based geometric estimation**

to estimate:

- image-plane pipeline center alignment,
- dominant pipeline orientation,
- directional alignment cues: `LEFT`, `STRAIGHT`, or `RIGHT`.

This project focuses on **lightweight embedded underwater perception**, not full robotic autonomy, SLAM, localization, or closed-loop AUV control.

---

## System Architecture

<p align="center">
  <img src="assets/architecture.jpg" alt="System Architecture" width="850"/>
</p>

The proposed pipeline operates in four sequential stages:

### 1. Image Acquisition

Monocular underwater RGB frames are captured and resized to **640 × 640 pixels**.

### 2. Instance Segmentation

A lightweight **YOLOv8n-seg** model predicts pixel-level pipeline masks directly from underwater imagery.

### 3. Deployment-Oriented Mask Reconstruction

ONNX outputs are reconstructed using:

- prototype-mask decoding,
- confidence filtering,
- bounding-box-guided cropping.

This stage preserves geometric consistency during deployment-oriented inference.

### 4. Geometry Extraction

PCA-based geometric estimation extracts:

- image-plane center position,
- orientation angle,
- directional alignment cue.

---

## Embedded Deployment

The trained YOLOv8n-seg model was exported to ONNX format and optimized for embedded inference on:

## Qualcomm RB3 Gen 2 / Dragonwing RB3 Gen 2

Deployment pipeline:

```text
PyTorch FP32
    ↓
ONNX Export
    ↓
INT8 Quantization
    ↓
Qualcomm AI Hub Compilation
    ↓
Hexagon NPU Deployment
```

The deployed INT8 model achieved:

| Metric | Value |
|---|---:|
| Inference latency | 8.8 ms |
| Throughput | 113.6 FPS |
| Processing unit | Qualcomm Hexagon NPU |
| NPU offload | 100% |

The PCA-based geometric extraction stage adds negligible computational overhead, with total post-processing below **0.5 ms/frame**.

---

## Datasets

## Training Dataset

The segmentation training dataset was derived from the publicly available **Cijevi underwater pipeline dataset** and manually re-annotated for instance segmentation.

### Characteristics

- 618 curated underwater pipeline images
- 3,090 images after augmentation
- polygon-based segmentation masks
- varying turbidity and illumination conditions
- multiple viewing angles
- 640 × 640 resolution

### Public Dataset

The segmentation-ready dataset is publicly available on Roboflow Universe:

**[Underwater Pipeline Dataset](https://universe.roboflow.com/hamzaghitri/pipeline-body)**

<p align="center">
  <a href="https://universe.roboflow.com/hamzaghitri/pipeline-body">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="Download Dataset from Roboflow"/>
  </a>
</p>

Representative underwater pipeline images under varying turbidity, illumination, and biofouling conditions:

<p align="center">
  <img src="assets/training_water_colors.JPG" alt="Training Dataset Visual Conditions" width="820"/>
</p>

Representative underwater pipeline images captured from different viewing angles:

<p align="center">
  <img src="assets/training_camera_angles.JPG" alt="Training Dataset Viewing Angles" width="820"/>
</p>

---

## External Evaluation Dataset A

Derived from publicly available **U.S. Coast Guard underwater inspection footage**.

Characteristics:

- severe turbidity,
- seabed clutter,
- low contrast,
- non-centered pipelines.

Public dataset:

<p align="center">
  <a href="https://universe.roboflow.com/hamzaghitri/underwater-pipeline-uscg">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="Download USCG Evaluation Dataset from Roboflow"/>
  </a>
</p>

<p align="center">
  <img src="assets/testA.jpg" alt="USCG Dataset" width="820"/>
</p>

---

## External Evaluation Dataset B

Derived from real-world ROV underwater inspection footage released by **Hibbard Inshore**.

Characteristics:

- heavy underwater degradation,
- biofouling,
- varying camera orientations,
- cluttered seabed conditions.

Public dataset:

<p align="center">
  <a href="https://universe.roboflow.com/hamzaghitri/underwater-pipeline-hibbard">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg" alt="Download Hibbard Evaluation Dataset from Roboflow"/>
  </a>
</p>

<p align="center">
  <img src="assets/testB.jpg" alt="Hibbard Dataset" width="820"/>
</p>

---

## Geometry Extraction

Navigation-relevant geometric information is extracted directly from the segmentation mask using PCA-based geometric estimation.

<p align="center">
  <img src="assets/pca_extraction.png" alt="PCA Extraction" width="850"/>
</p>

The pipeline estimates:

| Parameter | Description |
|---|---|
| `xc` | Image-plane pipeline center |
| `Δx` | Signed center offset |
| `α` | Dominant pipeline orientation angle |
| `direction` | `LEFT`, `STRAIGHT`, or `RIGHT` |

### Pipeline Center Estimation

```python
xc = mean(x_foreground_pixels)
```

### Center Offset

```python
delta_x = xc - image_width / 2
```

### Orientation Estimation

Principal Component Analysis is applied to foreground mask pixels:

```python
angle = atan2(dx, -dy)
```

### Direction Classification

```python
RIGHT      if α > +5°
LEFT       if α < -5°
STRAIGHT   otherwise
```

---

## Results

## Segmentation Performance

| Dataset | Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| Internal Test Set | FP32 | 0.986 | 1.000 | 0.995 | 0.946 |
| Internal Test Set | ONNX | 0.985 | 1.000 | 0.995 | 0.945 |
| External Test Set A | FP32 | 1.000 | 1.000 | 0.995 | 0.842 |
| External Test Set A | ONNX | 1.000 | 1.000 | 0.995 | 0.827 |
| External Test Set B | FP32 | 0.990 | 0.995 | 0.994 | 0.845 |
| External Test Set B | ONNX | 0.990 | 0.995 | 0.993 | 0.852 |

The results demonstrate robust cross-domain segmentation performance under challenging underwater conditions.

---

## Geometry Estimation Performance

| Dataset | Model | Center Error (px) | Angle Error (°) | Direction Accuracy |
|---|---|---:|---:|---:|
| Internal Test Set | FP32 | 3.12 | 1.00 | 99.52% |
| Internal Test Set | ONNX | 3.89 | 0.99 | 99.52% |
| External Test Set A | FP32 | 2.92 | 0.86 | 100.00% |
| External Test Set A | ONNX | 3.03 | 0.88 | 100.00% |
| External Test Set B | FP32 | 1.79 | 0.55 | 100.00% |
| External Test Set B | ONNX | 1.71 | 0.54 | 100.00% |

Across all evaluation datasets, the proposed framework maintained:

- sub-degree orientation estimation accuracy,
- stable directional consistency,
- minimal degradation after ONNX deployment.

---

## Embedded Deployment Performance

| Model | Hardware | Processing Unit | Latency | FPS |
|---|---|---|---:|---:|
| FP32 | NVIDIA Tesla T4 | GPU | 9.29 ms | 107.7 |
| INT8 | Qualcomm RB3 Gen 2 | Hexagon NPU | 8.80 ms | 113.6 |

The deployment results demonstrate that lightweight underwater perception can operate in real time on embedded AI hardware.


---

## Deployment Reconstruction Ablation

An ablation study evaluated the importance of deployment-side mask reconstruction during ONNX inference.

| Decoding Strategy | Center Error (px) | Angle Error (°) | Direction Accuracy |
|---|---:|---:|---:|
| No box-guided crop | 72.59 | 49.02 | 41.71% |
| Box-guided crop | 1.71 | 0.54 | 94.97% |

This demonstrates that:

> correct deployment-side mask reconstruction is critical for preserving downstream geometric consistency.

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/7amzaGH/Underwater-Pipeline-Geometric-Perception.git
cd Underwater-Pipeline-Geometric-Perception
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Model Weights

Place your:

- `best.pt`
- `best.onnx`

inside:

```text
models/
```

### 4. Run PyTorch Inference

```bash
python scripts/run_pytorch_inference.py \
    --model models/best.pt \
    --source demo/demo.mp4
```

### 5. Run ONNX Inference

```bash
python scripts/run_onnx_inference.py \
    --model models/best.onnx \
    --source demo/demo.mp4
```

### 6. Evaluate Geometry

```bash
python scripts/evaluate_geometry.py
```

---

## Usage in Python

```python
from src.geometry.pca import estimate_pipeline_geometry
from src.onnx_inference.inference import ONNXPipeline

pipeline = ONNXPipeline(
    model_path="models/best.onnx",
    conf_threshold=0.5
)

results = pipeline.predict(frame)

geometry = estimate_pipeline_geometry(results["mask"])

print(geometry)
```

---

## Notebook Demo

An interactive demonstration notebook is provided to reproduce the complete perception workflow and evaluation pipeline.

Notebook : [Pipeline Geometric Perception Demo](notebooks/Pipeline__Geometric_Perception_Demo.ipynb)

### Demonstrated Workflow

```text
Setup
    ↓
Dataset Download
    ↓
YOLOv8n-seg Inference
    ↓
Geometry Extraction
    ↓
Evaluation
    ↓
Visualization
```

The notebook demonstrates:

- repository setup and dependency installation,
- external evaluation dataset download from Roboflow,
- YOLOv8n-seg pipeline detection on underwater frames,
- PCA-based pipeline center and orientation estimation,
- center error, orientation error, and direction consistency evaluation,
- visualization of quantitative geometry analysis results.

### Perception Outputs

| Output | Description |
|---|---|
| Pipeline Center Position | Image-plane horizontal alignment |
| Lateral Offset | Displacement from image center |
| Pipeline Orientation | Dominant angular direction |
| Navigation Cue | `LEFT`, `STRAIGHT`, or `RIGHT` |

---

## C++ Embedded Geometry Runtime

In addition to the main Python implementation, this repository includes a lightweight **C++ embedded geometry runtime** that demonstrates how the geometry extraction stage can be implemented in a deployment-oriented structure.

The C++ component is provided as an embedded-oriented extension of the project and is not part of the experimental evaluation reported in the associated manuscript.

The C++ runtime focuses only on the post-processing stage after segmentation. It does **not** perform neural network inference, NPU execution, camera integration, SLAM, or robotic control.

```text
Segmentation mask / mask-point CSV
        ↓
C++ foreground point loading
        ↓
PCA-based geometry extraction
        ↓
center, angle, direction output
```

### Runtime Scope

| Included | Not Included |
|---|---|
| Mask-point CSV loading | Neural network inference |
| PCA-based center estimation | NPU runtime integration |
| Orientation estimation | Camera drivers |
| LEFT / STRAIGHT / RIGHT cue generation | Closed-loop robot control |
| CMake-based build structure | Full AUV software stack |

### Build and Run

```bash
cd cpp_runtime
mkdir build
cd build
cmake ..
cmake --build .
./pipeline_geometry_runtime ../examples/sample_mask_points.csv
```

Example output:

```text
Pipeline Geometry Runtime
-------------------------
Valid mask: true
Center x: 318.4 px
Center offset: -1.6 px
Orientation angle: 0.54 deg
Direction: STRAIGHT
```

---

## Repository Structure

```text
Underwater-Pipeline-Geometric-Perception/
│
├── assets/                 <- Figures, demo GIFs, and README visuals
├── models/
│   └── README.md           <- Models note
│
├── src/
│   ├── main_offline.py     <- Offline video perception pipeline
│   ├── main_live.py        <- Live camera deployment launcher
│   ├── detect.py           <- YOLOv8n inference
│   └── geometry.py         <- PCA-based geometric parameter extraction
│
├── scripts/                <- Inference and evaluation scripts
│
├── notebooks/
│   └── Pipeline__Geometric_Perception_Demo.ipynb
│
├── cpp_runtime/            <- C++ embedded geometry runtime
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── examples/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---
## Role within NeptuNet

<p align="center">
  <img src="assets/Neptunet_logo.png" alt="NeptuNet Logo" width="180"/>
</p>

This repository implements **Level 1** of the [NeptuNet](https://github.com/7amzaGH/NeptuNet-AUV-Intelligent-System) framework.

Within NeptuNet, this module provides continuous infrastructure context for underwater gas pipeline inspection.

| NeptuNet Level | Role |
|---|---|
| Level 1 — Pipeline Geometric Perception | Continuous pipeline context and image-plane geometry |
| Level 2 — Bubble-Based Early Warning | Activated when pipeline context is valid |
| Level 3 — Leak Confirmation | Activated when bubble activity becomes suspicious |

This repository is intentionally maintained as an independent research artifact while also serving as the Level 1 component of the full NeptuNet ecosystem.

---

# Citation

If you use this work in your research, please cite:

```bibtex
@misc{ghitri2026pipeline,
  title        = {Lightweight Underwater Pipeline Geometric Perception for Embedded AI Deployment},
  author       = {Ghitri, Hamza and Belgrana, Fatima Zohra},
  year         = {2026},
  note         = {Manuscript under review},
  howpublished = {\url{https://github.com/7amzaGH/Underwater-Pipeline-Geometric-Perception}}
}
```
The citation will be updated after formal publication with the official venue, DOI, and bibliographic metadata.


---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai)
- [Qualcomm AI Hub](https://aihub.qualcomm.com)
- [Cijevi Underwater Pipeline Dataset](https://universe.roboflow.com/boris-gasparovic/cijevi)
- U.S. Coast Guard underwater inspection footage
- Hibbard Inshore underwater ROV inspection footage

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Lightweight embedded underwater pipeline perception for geometric inspection awareness.</sub>
</div>
