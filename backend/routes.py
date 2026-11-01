
import shutil
import uuid
import logging
import sys

from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.responses import FileResponse

from backend.schemas import (
    GenerationResponse,
    HealthResponse
)

# =========================================
# LOGGING CONFIGURATION
# =========================================

logger = logging.getLogger("uvicorn.error")


# =========================================
# INSTANTMESH IMPORT PATH
# =========================================

INSTANTMESH_DIR = Path("/content/InstantMesh")

if str(INSTANTMESH_DIR) not in sys.path:
    sys.path.insert(0, str(INSTANTMESH_DIR))

# Transformers 5.x compatibility for InstantMesh
import torch
import transformers.pytorch_utils as pytorch_utils

def find_pruneable_heads_and_indices(
    heads, n_heads, head_size, already_pruned_heads
):
    mask = torch.ones(n_heads, head_size)

    heads = set(heads) - already_pruned_heads

    for head in heads:
        head -= sum(1 for h in already_pruned_heads if h < head)
        mask[head] = 0

    index = torch.arange(mask.numel())[mask.reshape(-1).bool()]
    return heads, index.long()

pytorch_utils.find_pruneable_heads_and_indices = (
    find_pruneable_heads_and_indices
)

from src.models.encoder.dino import ViTModel

def compatible_get_head_mask(self, head_mask, num_hidden_layers):
    if head_mask is None:
        return [None] * num_hidden_layers

    return self._convert_head_mask_to_5d(
        head_mask, num_hidden_layers
    )

ViTModel.get_head_mask = compatible_get_head_mask

# Import pipeline only after applying compatibility patches
from services.pipeline import Sketch2ShapePipeline


# =========================================
# ROUTER CONFIGURATION
# =========================================

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "examples" / "input_sketches"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# INITIALIZE PIPELINE
# =========================================

pipeline = Sketch2ShapePipeline(
    base_dir=BASE_DIR,
    instantmesh_repo=INSTANTMESH_DIR,
    grid_res=64
)


# =========================================
# HEALTH CHECK
# =========================================

@router.get("/health", response_model=HealthResponse)
async def health_check():

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HealthResponse(
        status="healthy",
        device=device,
        message="Sketch2Shape AI backend is running."
    )


# =========================================
# GENERATE 3D MODEL
# =========================================

@router.post(
    "/generate",
    response_model=GenerationResponse
)
async def generate_3d_model(
    image: UploadFile = File(...),
    input_type: str = Form(...),
    object_name: str = Form("object"),
    style: str = Form("photorealistic"),
    material: str = Form("not specified"),
    seed: int = Form(42)
):
    """
    Upload a sketch/photo and generate a 3D model.
    """

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg"
    }

    # Validate uploaded image
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are supported."
        )

    # Validate input type
    if input_type not in ["sketch", "photo"]:
        raise HTTPException(
            status_code=400,
            detail="input_type must be 'sketch' or 'photo'."
        )

    # Unique request identifier
    request_id = uuid.uuid4().hex[:12]

    extension = (
        ".png"
        if image.content_type == "image/png"
        else ".jpg"
    )

    input_path = INPUT_DIR / f"{request_id}{extension}"

    try:

        # Save uploaded image
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(
                image.file,
                buffer
            )

        logger.info(
            f"[{request_id}] Starting 3D generation."
        )

        # Run complete ML pipeline
        result = pipeline.process(
            input_path=input_path,
            input_type=input_type,
            object_name=object_name,
            style=style,
            material=material,
            seed=seed
        )

        logger.info(
            f"[{request_id}] 3D generation completed."
        )

        # Return generated model details
        return GenerationResponse(
            success=True,
            message="3D model generated successfully.",
            generated_image=result["generated_image"],
            processed_image=result["processed_image"],
            glb_path=result["export"]["glb_path"],
            obj_path=result["export"]["obj_path"],
            validation=result["validation"]
        )

    except HTTPException:
        raise

    except Exception as error:

        # Print complete traceback in backend logs
        logger.exception(
            f"[{request_id}] Sketch2Shape generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"3D generation failed: "
                f"{type(error).__name__}: {str(error)}"
            )
        )

    finally:

        # Close uploaded image
        image.file.close()

        # Remove temporary uploaded input
        if input_path.exists():
            input_path.unlink(missing_ok=True)


# =========================================
# DOWNLOAD GENERATED MODEL
# =========================================

@router.get("/download/{filename}")
async def download_model(filename: str):
    """
    Download generated GLB or OBJ files.
    """

    # Prevent directory traversal
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    # Allow only supported 3D formats
    if not filename.lower().endswith(
        (".glb", ".obj")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only GLB and OBJ downloads are allowed."
        )

    file_path = OUTPUT_DIR / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Model file not found."
        )

    return FileResponse(
        path=file_path,
        filename=filename
    )
