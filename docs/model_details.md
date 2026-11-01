
# Model Details

## 1. Stable Diffusion 1.5 + ControlNet

**Purpose:** Sketch-guided image generation.

- Uses the input sketch as structural guidance.
- Generates a visually detailed image.
- Helps preserve the main characteristics of the sketch.

## 2. Zero123++

**Purpose:** Multi-view image generation.

- Takes the generated image as input.
- Generates multiple viewpoints of the object.
- Provides visual information for 3D reconstruction.

## 3. InstantMesh

**Purpose:** 3D mesh reconstruction.

- Uses multi-view image information.
- Reconstructs a 3D object.
- Generates mesh geometry.
- Supports exporting the reconstructed model.

## Model Download and Setup

Pretrained model weights are not included in this repository.

They must be downloaded from their respective model repositories before inference.

The model loader handles model initialization and loading.

Refer to `services/model_loader.py` and `configs/config.py` for implementation details.

## Hardware

Recommended environment:
- Google Colab with NVIDIA T4 GPU.
- CUDA-compatible PyTorch installation.
- Sufficient disk space for pretrained model weights.
