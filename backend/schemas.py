
from typing import Literal, Optional
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """
    Request schema for Sketch2Shape 3D generation.
    """

    input_type: Literal["sketch", "photo"] = Field(
        ...,
        description="Input image type: sketch or photo"
    )

    object_name: str = Field(
        default="object",
        min_length=1,
        max_length=100,
        description="Name of the object"
    )

    style: str = Field(
        default="photorealistic",
        description="Visual style for sketch generation"
    )

    material: str = Field(
        default="not specified",
        description="Desired object material"
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1
    )

    num_inference_steps: int = Field(
        default=75,
        ge=20,
        le=100
    )


class GenerationResponse(BaseModel):
    """
    Response schema after successful 3D generation.
    """

    success: bool

    message: str

    generated_image: Optional[str] = None

    processed_image: Optional[str] = None

    glb_path: Optional[str] = None

    obj_path: Optional[str] = None

    validation: Optional[dict] = None

    error: Optional[str] = None


class HealthResponse(BaseModel):
    """
    API health-check response.
    """

    status: str

    device: str

    message: str
