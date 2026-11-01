
import gc
import torch

from pathlib import Path
from PIL import Image

from ml.preprocessing import (
    preprocess_sketch,
    remove_background
)

from ml.image_generation import ImageGenerator
from ml.reconstruction import Reconstruction
from ml.mesh_validation import MeshValidator

from services.model_loader import ModelLoader


class Sketch2ShapePipeline:
    """
    End-to-end Sketch2Shape AI pipeline.
    Supports sketch-to-3D and photo-to-3D.
    """

    def __init__(
        self,
        base_dir,
        instantmesh_repo=None,
        grid_res=80
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = ModelLoader(
            base_dir=base_dir,
            instantmesh_repo=instantmesh_repo
        )

        self.grid_res = grid_res

    def process(
        self,
        input_path,
        input_type,
        object_name="object",
        style="photorealistic",
        material="not specified",
        seed=42
    ):
        """
        Run complete image-to-3D generation.

        input_type: 'sketch' or 'photo'
        """

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {input_path}"
            )

        if input_type not in ["sketch", "photo"]:
            raise ValueError(
                "input_type must be 'sketch' or 'photo'"
            )

        print("Starting Sketch2Shape pipeline...")

        # ----------------------------------
        # 1. Input preprocessing
        # ----------------------------------

        if input_type == "sketch":

            print("Preprocessing sketch...")

            sketch_path = self.output_dir / "processed_sketch.png"

            processed_sketch = preprocess_sketch(
                str(input_path),
                str(sketch_path)
            )

            # ----------------------------------
            # 2. Sketch-to-image generation
            # ----------------------------------

            print("Generating realistic image using ControlNet...")

            controlnet_pipeline = self.loader.load_controlnet()

            image_generator = ImageGenerator.__new__(ImageGenerator)
            image_generator.device = self.loader.device
            image_generator.pipeline = controlnet_pipeline

            generated_image = image_generator.generate(
                sketch=processed_sketch,
                object_name=object_name,
                style=style,
                material=material,
                seed=seed
            )

            generated_path = self.output_dir / "generated_image.png"
            generated_image.save(generated_path)

            del image_generator
            self.loader.unload_controlnet()

            source_image_path = generated_path

        else:
            print("Using original photograph...")
            source_image_path = input_path

        # ----------------------------------
        # 3. Background removal
        # ----------------------------------

        print("Removing background...")

        processed_path = self.output_dir / "processed_object.png"

        processed_image = remove_background(
            str(source_image_path),
            str(processed_path)
        )

        # ----------------------------------
        # 4. Zero123++ multi-view generation
        # ----------------------------------

        print("Generating six viewpoints...")

        zero123_pipeline = self.loader.load_zero123()

        reconstruction = Reconstruction(
            zero123_pipeline=zero123_pipeline,
            instantmesh_model=None,
            device=self.loader.device
        )

        views = reconstruction.generate_views(
            processed_image,
            num_inference_steps=75,
            seed=seed
        )

        views_dir = self.output_dir / "zero123_views"
        views_dir.mkdir(exist_ok=True)

        for index, view in enumerate(views, start=1):
            view.save(views_dir / f"view_{index}.png")

        del reconstruction
        self.loader.unload_zero123()

        # ----------------------------------
        # 5. InstantMesh reconstruction
        # ----------------------------------

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Reconstructing 3D geometry...")

        instantmesh_model = self.loader.load_instantmesh(
            grid_res=self.grid_res
        )

        reconstruction = Reconstruction(
            zero123_pipeline=None,
            instantmesh_model=instantmesh_model,
            device=self.loader.device
        )

        vertices, faces, colors = reconstruction.generate_mesh(
            views=views,
            camera_radius=4.0
        )

        del reconstruction
        del instantmesh_model

        self.loader.unload_instantmesh()

        # ----------------------------------
        # 6. Mesh cleaning and validation
        # ----------------------------------

        print("Cleaning generated 3D mesh...")

        validator = MeshValidator(
            vertices=vertices,
            faces=faces,
            colors=colors
        )

        # Remove tiny disconnected artifacts
        validator.remove_small_components(
            min_face_count=100
        )

        print("Mesh cleaning completed.")

        # Validate cleaned mesh
        print("Validating cleaned 3D mesh...")

        validation = validator.validate()

        print("Mesh validation results:")

        for key, value in validation.items():
            print(f"{key}: {value}")

        # ----------------------------------
        # 7. Export cleaned 3D model
        # ----------------------------------

        print("Exporting cleaned 3D model...")

        export_info = validator.export(
            output_dir=str(self.output_dir),
            filename="generated_model"
        )

        print("GLB:", export_info["glb_path"])
        print("OBJ:", export_info["obj_path"])

        # ----------------------------------
        # 8. Memory cleanup
        # ----------------------------------

        del vertices, faces, colors, views, processed_image

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Sketch2Shape pipeline completed!")

        return {
            "validation": validation,
            "export": export_info,
            "generated_image": str(source_image_path),
            "processed_image": str(processed_path)
        }
