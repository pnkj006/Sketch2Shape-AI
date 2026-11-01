
import os
import tempfile
import requests
import gradio as gr

from pathlib import Path
from PIL import Image


# =========================================
# CONFIGURATION
# =========================================

API_BASE_URL = os.getenv(
    "SKETCH2SHAPE_API",
    "http://127.0.0.1:8000"
)


# =========================================
# IMAGE PROCESSING
# =========================================

def get_image_path(image_data):
    """
    Extract an image filepath from Gradio components.
    Supports Sketchpad and Image upload.
    """

    if image_data is None:
        return None

    if isinstance(image_data, dict):
        image_data = (
            image_data.get("composite")
            or image_data.get("background")
        )

    if isinstance(image_data, Image.Image):
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        image_data.convert("RGB").save(temp_path)
        return str(temp_path)

    if isinstance(image_data, str):
        return image_data

    return None


# =========================================
# GENERATION FUNCTION
# =========================================

def generate_3d(
    canvas_image,
    uploaded_image,
    input_type,
    object_name,
    description,
    style,
    material
):

    # Uploaded image takes priority if provided
    if uploaded_image is not None:
        image_path = get_image_path(uploaded_image)
    else:
        image_path = get_image_path(canvas_image)

    if not image_path or not os.path.exists(image_path):
        raise gr.Error(
            "Please draw something on the canvas or upload an image."
        )

    if input_type not in ["sketch", "photo"]:
        raise gr.Error("Please select an input type.")

    # Combine object name and optional description
    combined_object_name = (
        object_name.strip()
        if object_name and object_name.strip()
        else "object"
    )

    if description and description.strip():
        combined_object_name += ". " + description.strip()

    endpoint = f"{API_BASE_URL}/api/generate"

    try:

        # Convert input to RGB PNG for consistent upload
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
            ) as temp_file:
                upload_path = Path(temp_file.name)

            img.save(upload_path, format="PNG")

        # Send image and parameters to backend
        with open(upload_path, "rb") as image_file:

            files = {
                "image": (
                    "input.png",
                    image_file,
                    "image/png"
                )
            }

            data = {
                "input_type": input_type,
                "object_name": combined_object_name,
                "style": style,
                "material": material,
                "seed": 42
            }

            response = requests.post(
                endpoint,
                files=files,
                data=data,
                timeout=1800
            )

        response.raise_for_status()

        result = response.json()

        if not result.get("success"):
            raise gr.Error(
                result.get("message", "Generation failed.")
            )

        glb_path = result.get("glb_path")
        obj_path = result.get("obj_path")

        if not glb_path or not obj_path:
            raise gr.Error(
                "Generated model paths are missing."
            )

        # =================================
        # DOWNLOAD GENERATED MODEL FILES
        # =================================

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="sketch2shape_output_"
            )
        )

        downloaded_files = []

        for remote_path in [glb_path, obj_path]:

            filename = Path(remote_path).name

            download_url = (
                f"{API_BASE_URL}/api/download/{filename}"
            )

            file_response = requests.get(
                download_url,
                timeout=120
            )

            file_response.raise_for_status()

            local_path = temp_dir / filename

            local_path.write_bytes(
                file_response.content
            )

            downloaded_files.append(str(local_path))

        glb_file, obj_file = downloaded_files

        # =================================
        # MESH VALIDATION
        # =================================

        validation = result.get("validation", {})

        validation_text = "\n".join([
            f"Vertices: {validation.get('vertex_count', 'N/A')}",
            f"Faces: {validation.get('face_count', 'N/A')}",
            f"Watertight: {validation.get('watertight', 'N/A')}",
            f"Winding consistent: {validation.get('winding_consistent', 'N/A')}",
            f"Surface area: {validation.get('surface_area', 'N/A')}",
            f"Volume: {validation.get('volume', 'N/A')}"
        ])

        return (
            "3D model generated successfully!",
            glb_file,
            glb_file,
            obj_file,
            validation_text
        )

    except gr.Error:
        raise

    except requests.RequestException as error:
        raise gr.Error(
            f"Backend connection failed: {error}"
        )

    except Exception as error:
        raise gr.Error(
            f"Generation failed: {error}"
        )


# =========================================
# GRADIO FRONTEND
# =========================================

with gr.Blocks(
    title="Sketch2Shape AI"
) as demo:

    gr.Markdown(
        """
        # Sketch2Shape AI

        ### Turn your ideas into 3D models.

        Draw your idea or upload an existing image.
        Add an optional description and generate an interactive 3D model.
        """
    )

    with gr.Row():

        # =================================
        # LEFT COLUMN: INPUT
        # =================================

        with gr.Column(scale=1):

            gr.Markdown("## 1. Create Your Input")

            input_type = gr.Radio(
                choices=["sketch", "photo"],
                value="sketch",
                label="Input Type"
            )

            gr.Markdown("### Draw your sketch")

            canvas_image = gr.Sketchpad(
                type="filepath",
                label="Drawing Canvas",
                height=450,
                width=450,
                brush=gr.Brush(
                    default_size=5,
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed"
                )
            )

            gr.Markdown(
                "### Or upload an existing image"
            )

            uploaded_image = gr.Image(
                type="filepath",
                sources=["upload"],
                label="Upload Sketch / Photograph",
                height=250
            )

            gr.Markdown("## 2. Describe Your Idea")

            object_name = gr.Textbox(
                label="Object Name",
                placeholder="e.g. Wooden dining chair",
                value="object"
            )

            description = gr.Textbox(
                label="Optional Description",
                placeholder=(
                    "Describe the shape, structure, "
                    "dimensions, details or appearance "
                    "of your desired 3D model..."
                ),
                lines=4
            )

            gr.Markdown("## 3. Generation Settings")

            style = gr.Dropdown(
                choices=[
                    "photorealistic",
                    "realistic",
                    "cartoon",
                    "low-poly"
                ],
                value="photorealistic",
                label="Visual Style"
            )

            material = gr.Textbox(
                label="Material",
                placeholder="e.g. Natural oak wood",
                value="not specified"
            )

            generate_button = gr.Button(
                "Generate 3D Model",
                variant="primary",
                size="lg"
            )

        # =================================
        # RIGHT COLUMN: OUTPUT
        # =================================

        with gr.Column(scale=1):

            gr.Markdown("## 4. Generated 3D Model")

            status_output = gr.Textbox(
                label="Generation Status",
                interactive=False
            )

            model_preview = gr.Model3D(
                label="Interactive 3D Preview",
                clear_color=(0.1, 0.1, 0.1, 1)
            )

            gr.Markdown("### Download Your Model")

            with gr.Row():

                glb_download = gr.File(
                    label="Download GLB"
                )

                obj_download = gr.File(
                    label="Download OBJ"
                )

            with gr.Accordion(
                "Mesh Validation Details",
                open=False
            ):

                validation_output = gr.Textbox(
                    label="Validation Results",
                    lines=7,
                    interactive=False
                )

    # =================================
    # GENERATE BUTTON EVENT
    # =================================

    generate_button.click(
        fn=generate_3d,
        inputs=[
            canvas_image,
            uploaded_image,
            input_type,
            object_name,
            description,
            style,
            material
        ],
        outputs=[
            status_output,
            model_preview,
            glb_download,
            obj_download,
            validation_output
        ]
    )


# =========================================
# LAUNCH APPLICATION
# =========================================

if __name__ == "__main__":

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7861,
        theme=gr.themes.Soft()
    )
