
# Sketch2Shape AI

Transform 2D sketches into visually detailed 3D models using Generative AI.

## Overview

Sketch2Shape AI is a sketch-to-3D generation system that converts hand-drawn sketches or uploaded images into 3D models.

It combines sketch-guided image generation, multi-view synthesis and 3D reconstruction.

## Features

- In-app sketch drawing.
- Sketch image upload.
- Optional text prompts.
- Sketch-guided image generation.
- Multi-view generation.
- 3D mesh reconstruction.
- Mesh validation.
- GLB and OBJ export.
- Interactive 3D preview.

## Technology Stack

- Python
- PyTorch
- Stable Diffusion 1.5
- ControlNet
- Zero123++
- InstantMesh
- OpenCV
- Trimesh
- FastAPI
- Gradio

## Project Structure

Refer to the project directory for the complete source organization.

## Installation

Clone the repository:

```bash
git clone https://github.com/pnkj006/Sketch2Shape-AI.git
cd Sketch2Shape-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install compatible PyTorch, CUDA and nvdiffrast dependencies as described in the setup notebook.

## Model Setup

Pretrained model weights are not included in this repository.

Download and configure the required models:

- Stable Diffusion 1.5 + ControlNet
- Zero123++
- InstantMesh

Refer to `docs/model_details.md` and
`notebooks/sketch2shape_setup.ipynb`.

## Running the Project

Recommended environment: Google Colab with an NVIDIA T4 GPU.

Open:

`notebooks/sketch2shape_deployment.ipynb`

Follow the notebook instructions to initialize the models and launch the application.

## Output

Generated models are available in:

`outputs/`

Supported formats:

- GLB
- OBJ

## Examples

Sample input sketches are available in:

`examples/input_sketches/`

A generated sample model is available in:

`outputs/`

## Documentation

- `docs/architecture.md`
- `docs/model_details.md`
- `docs/limitations.md`

## Limitations

Refer to `docs/limitations.md`.

## Hardware

Recommended: NVIDIA T4 GPU or equivalent CUDA-compatible GPU.


