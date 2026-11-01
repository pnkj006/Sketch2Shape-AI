
# System Architecture

## Overview

Sketch2Shape AI converts a single-view 2D sketch into a textured 3D model.

## Pipeline

1. User draws or uploads a sketch.
2. Input preprocessing is performed.
3. ControlNet generates a sketch-guided image.
4. Background removal is applied where required.
5. Zero123++ generates multiple viewpoints.
6. InstantMesh reconstructs the 3D geometry.
7. Mesh validation and cleanup are performed.
8. The final model is exported as GLB or OBJ.

## Main Components

| Component | Responsibility |
|---|---|
| frontend/ | User interface |
| backend/ | API and request handling |
| ml/ | Image processing and reconstruction |
| services/ | Model loading and pipeline orchestration |
| configs/ | Configuration management |
| notebooks/ | Setup, testing and deployment |
| tests/ | Preprocessing and mesh export tests |
| examples/ | Sample input and generated output |
| outputs/ | Generated 3D models |
