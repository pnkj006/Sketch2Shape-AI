import sys
import torch
import numpy as np

from pathlib import Path
from PIL import Image

INSTANTMESH_DIR = Path("/content/InstantMesh")

if str(INSTANTMESH_DIR) not in sys.path:
    sys.path.insert(0, str(INSTANTMESH_DIR))

from src.utils.camera_util import get_zero123plus_input_cameras

class Reconstruction:
    """
    Image-to-3D reconstruction using Zero123++
    and InstantMesh.
    """

    def __init__(
        self,
        zero123_pipeline,
        instantmesh_model,
        device=None
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.zero123_pipeline = zero123_pipeline
        self.instantmesh_model = instantmesh_model

    def generate_views(
        self,
        image,
        num_inference_steps=75,
        seed=42
    ):
        """
        Generate six views using Zero123++.

        Returns six individual PIL images.
        """

        image = image.convert("RGBA")

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        with torch.inference_mode():
            result = self.zero123_pipeline(
                image,
                num_inference_steps=num_inference_steps,
                generator=generator
            )

        contact_sheet = result.images[0]

        width, height = contact_sheet.size

        if width % 2 != 0 or height % 3 != 0:
            raise ValueError(
                "Unexpected Zero123++ contact sheet dimensions."
            )

        view_width = width // 2
        view_height = height // 3

        views = []

        for row in range(3):
            for col in range(2):

                left = col * view_width
                top = row * view_height

                view = contact_sheet.crop((
                    left,
                    top,
                    left + view_width,
                    top + view_height
                ))

                views.append(view.convert("RGB"))

        if len(views) != 6:
            raise RuntimeError(
                "Zero123++ did not produce six viewpoints."
            )

        return views

    def generate_mesh(
        self,
        views,
        camera_radius=4.0,
        use_texture_map=False
    ):
        """
        Reconstruct a 3D mesh from six generated views.

        Returns:
            vertices
            faces
            colors
        """

        if len(views) != 6:
            raise ValueError(
                "InstantMesh requires exactly six viewpoints."
            )

        # Convert PIL images to normalized tensors
        images = np.stack([
            np.asarray(
                view.resize((320, 320)),
                dtype=np.float32
            ) / 255.0
            for view in views
        ])

        images = torch.from_numpy(images)

        images = images.permute(
            0, 3, 1, 2
        ).contiguous()

        images = images.unsqueeze(0).to(self.device)

        # Generate official Zero123++ camera parameters
        cameras = get_zero123plus_input_cameras(
            batch_size=1,
            radius=camera_radius
        ).to(self.device)

        # Generate triplanes
        with torch.inference_mode():
            planes = self.instantmesh_model.forward_planes(
                images,
                cameras
            )

            # Extract mesh
            vertices, faces, colors = (
                self.instantmesh_model.extract_mesh(
                    planes,
                    use_texture_map=use_texture_map
                )
            )

        return vertices, faces, colors

    def reconstruct(
        self,
        image,
        num_inference_steps=75,
        seed=42,
        camera_radius=4.0
    ):
        """
        Complete image-to-3D reconstruction.

        Returns:
            views, vertices, faces, colors
        """

        views = self.generate_views(
            image,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

        vertices, faces, colors = self.generate_mesh(
            views,
            camera_radius=camera_radius
        )

        return {
            "views": views,
            "vertices": vertices,
            "faces": faces,
            "colors": colors
        }
