# Sketch2Shape AI
### From 2D Sketches to 3D Geometry

**Turn your sketches into textured 3D models using Generative AI.**

Sketch2Shape AI is a Generative AI-powered application that transforms hand-drawn sketches or uploaded images into 3D models. It combines sketch-conditioned image generation, multi-view synthesis, and 3D reconstruction to create downloadable 3D assets.

**Hackathon:** TechZephyr 4.0 — IIT Bhubaneswar  
**Team:** Team Sky  
**Team Members:** Amisha Tripathy, Pankaj Maity, Sanjib Kumar Mandal

**GitHub:** https://github.com/pnkj006/Sketch2Shape-AI

---

## Features

- Interactive sketch canvas with brush, eraser, and reset controls.
- Upload an existing sketch or image.
- Optional text prompt for additional generation guidance.
- Sketch-guided image generation using ControlNet.
- Multi-view image synthesis using Zero123++.
- 3D reconstruction using InstantMesh.
- Mesh validation and cleanup.
- Interactive 3D model preview.
- Export reconstructed models in **GLB and OBJ formats**.

## System Architecture

```text
        User Sketch / Uploaded Image
                    |
                    v
             Preprocessing
                    |
                    v
       Stable Diffusion 1.5 + ControlNet
                    |
                    v
          Generated Object Image
                    |
                    v
               Zero123++
                    |
                    v
          Multiple Object Views
                    |
                    v
               InstantMesh
                    |
                    v
          3D Mesh Reconstruction
                    |
                    v
          Mesh Validation / Cleanup
                    |
                    v
             GLB / OBJ Export
```

## AI Models Used

| Model | Purpose |
|---|---|
| Stable Diffusion 1.5 | Image generation |
| ControlNet Scribble | Sketch-guided generation |
| Zero123++ | Multi-view synthesis |
| InstantMesh | 3D reconstruction |

The project integrates pretrained models rather than training a new foundation model from scratch.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Image Processing:** OpenCV, Pillow
- **Generative AI:** Diffusers, ControlNet
- **3D Processing:** InstantMesh, Trimesh
- **Backend:** FastAPI
- **Interface:** Gradio
- **Environment:** Google Colab, NVIDIA T4 GPU

## Project Structure

```text
Sketch2Shape-AI/
├── backend/
├── ml/
├── services/
├── frontend/
├── configs/
├── notebooks/
├── examples/
├── outputs/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Processing Workflow

1. **Input:** Draw a sketch or upload an image.
2. **Preprocessing:** Prepare the input for model inference.
3. **Image Generation:** Generate a detailed image using Stable Diffusion and ControlNet.
4. **Background Processing:** Isolate the object where required.
5. **Multi-view Synthesis:** Generate additional viewpoints using Zero123++.
6. **3D Reconstruction:** Reconstruct geometry using InstantMesh.
7. **Mesh Validation:** Inspect and clean the generated mesh.
8. **Export:** Preview and download the model in GLB or OBJ format.

## Demonstrated Results

The following results were obtained during project testing:

| Metric | Generated Model |
|---|---:|
| GLB file size | 321.80 KB |
| OBJ file size | 822.85 KB |
| Vertices | 8,216 |
| Faces | 16,424 |
| Watertight | Yes |

A separate mesh-validation test also produced a watertight mesh with 11,808 vertices and 23,612 faces.

These are observed test results, not guaranteed results for every input sketch.

## Setup and Execution

### Recommended Environment

- Google Colab
- NVIDIA T4 GPU or compatible CUDA GPU
- Python environment with compatible PyTorch, CUDA, and model dependencies

### Clone the repository

```bash
git clone https://github.com/pnkj006/Sketch2Shape-AI.git
cd Sketch2Shape-AI
```

### Model Setup

The pretrained model weights are downloaded separately and are not included in the GitHub repository.

Use the project notebooks:

- `notebooks/sketch2shape_setup.ipynb`
- `notebooks/sketch2shape_deployment.ipynb`
- `notebooks/model_testing.ipynb`

Follow the notebook setup steps to configure model dependencies, CUDA, and InstantMesh.

**Note:** Model initialization and inference require substantial GPU resources. Dependency versions must remain compatible with the Colab PyTorch and CUDA environment.

## Hardware and Performance

The project was tested using an NVIDIA T4 GPU in Google Colab. InstantMesh was configured with `grid_res=64` to manage GPU memory.

Generation time and peak GPU memory have not been benchmarked consistently, so no fixed performance figures are claimed.

## Limitations

- Output quality depends on sketch clarity and structure.
- Unseen surfaces are inferred by generative models.
- Complex shapes and fine geometric details may not reconstruct perfectly.
- Multi-view inconsistencies can affect the final geometry.
- GPU memory and model initialization requirements can affect execution.
- Generated meshes are intended for visual and creative use, not precision engineering or guaranteed 3D printing.

## Future Improvements

- Improve multi-view consistency.
- Enhance mesh refinement and topology.
- Optimize inference speed and GPU memory usage.
- Test across more object categories.
- Improve reconstruction quality for complex sketches.

## Acknowledgements

This project builds upon pretrained models and open-source research:

- Stable Diffusion and ControlNet
- Zero123++
- InstantMesh

All underlying pretrained models remain the work of their respective creators.

---

**Sketch2Shape AI — Turning simple sketches into 3D possibilities.**
