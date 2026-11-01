
import torch

from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler
)


class ImageGenerator:
    """
    Generate realistic images from 2D sketches
    using Stable Diffusion and Scribble ControlNet.
    """

    def __init__(
        self,
        controlnet_id="lllyasviel/sd-controlnet-scribble",
        sd_model_id="runwayml/stable-diffusion-v1-5"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=dtype
        )

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()

        self.pipeline = self.pipeline.to(self.device)

    @staticmethod
    def generate_prompt(
        object_name,
        style="photorealistic",
        material="not specified",
        background="clean white background"
    ):
        """
        Generate a prompt using a local template.
        """

        prompt = f"""
        A {style} {object_name},
        based on the exact input sketch,
        preserve the original structure and proportions,
        preserve all visible components and connections,
        {material},
        realistic three-dimensional appearance,
        professional product photography,
        {background},
        single isolated object, centered composition.
        """

        negative_prompt = """
        extra parts, missing components, distorted proportions,
        duplicate objects, floating parts, disconnected components,
        unrealistic geometry, blurry, low quality,
        text, watermark.
        """

        return prompt.strip(), negative_prompt.strip()

    def generate(
        self,
        sketch,
        object_name,
        style="photorealistic",
        material="not specified",
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=42
    ):
        """
        Generate a realistic image from a sketch.

        sketch: PIL Image
        object_name: User-provided object description
        """

        prompt, negative_prompt = self.generate_prompt(
            object_name=object_name,
            style=style,
            material=material
        )

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sketch.convert("RGB"),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=512,
                width=512
            )

        return result.images[0]
