# Pipeline C++ Embedded Geometry Runtime

This folder contains a lightweight C++ runtime prototype for the **Underwater Pipeline Geometric Perception** repository.

The goal is to demonstrate how the pipeline geometry extraction stage can be expressed in an embedded-friendly C++ structure.

This component does **not** perform neural network inference, ONNX inference, NPU execution, camera integration, underwater robot control, SLAM, or full onboard AUV deployment.

It is intentionally limited to the **post-processing stage** after segmentation.

---

## Purpose

The runtime reads foreground mask points exported from a segmentation mask and computes compact image-plane pipeline geometry:

```text
Segmentation mask / mask-point CSV
        ↓
C++ foreground point loader
        ↓
PCA-based geometry extraction
        ↓
center offset, orientation angle, direction cue
```

This is useful for showing how the lightweight geometric reasoning stage can be moved from Python experimentation toward an embedded-style C++ implementation.

---

## Runtime Outputs

| Output | Meaning |
|---|---|
| `valid` | Whether enough foreground points exist for stable PCA estimation |
| `center_x` | Mean horizontal coordinate of foreground mask points |
| `center_y` | Mean vertical coordinate of foreground mask points |
| `center_offset` | Signed horizontal offset from image center |
| `orientation_angle` | Dominant pipeline orientation angle in degrees |
| `direction` | Directional cue: `LEFT`, `STRAIGHT`, or `RIGHT` |

---

## Folder Structure

```text
cpp_runtime/
│
├── README.md
├── CMakeLists.txt
│
├── include/
│   └── pipeline_geometry.hpp
│
├── src/
│   ├── pipeline_geometry.cpp
│   └── main.cpp
│
└── examples/
    └── sample_mask_points.csv
```

---

## Build

From inside `cpp_runtime/`:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Run

Run the included example:

```bash
./pipeline_geometry_runtime ../examples/sample_mask_points.csv
```

Run with custom image size and direction threshold:

```bash
./pipeline_geometry_runtime ../examples/sample_mask_points.csv 640 640 5.0
```

---

## Input Format

The runtime reads a simple CSV file containing foreground mask pixel coordinates.

```csv
x,y
308.40,80
313.40,80
318.40,80
323.40,80
328.40,80
...
```

Each row represents one foreground pixel or sampled foreground point from the predicted pipeline segmentation mask.

The example CSV is intentionally small and human-readable. In a real pipeline, these points would be generated from the binary segmentation mask produced by the Python or ONNX inference stage.

---

## Example Output

```text
Input file: ../examples/sample_mask_points.csv
Image size: 640x640
Direction threshold: 5 deg

Pipeline Geometry Runtime
-------------------------
Valid mask: true
Foreground points: 305
Center x: 318.40 px
Center y: 320.00 px
Center offset: -1.60 px
Orientation angle: 0.54 deg
Direction: STRAIGHT
```

---

## Scope Note

This C++ runtime is an **embedded-oriented extension**, not the main experimental contribution of the repository.

It does not claim:

- real-time onboard AUV deployment,
- closed-loop control,
- SLAM or underwater localization,
- real neural network inference,
- direct NPU runtime execution,
- physical underwater deployment validation.

Its purpose is to show how the mask-to-geometry stage can be written as a clean C++ component suitable for future embedded integration.

---

## Relation to the Python Pipeline

The Python pipeline is used for training, inference, visualization, and quantitative evaluation.

The C++ runtime focuses on the lightweight geometric post-processing stage.

```text
Python / ONNX pipeline
        ↓
Segmentation mask generation and evaluation

C++ runtime skeleton
        ↓
Embedded-style PCA geometry extraction prototype
```
