
import gc
import sys
import torch

from pathlib import Path
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

from omegaconf import OmegaConf
from src.utils.train_util import instantiate_from_config


class ModelLoader:
    """
    Centralized model loader for Sketch2Shape AI.

    Models are loaded on demand to reduce GPU memory usage.
    """

    def __init__(self, base_dir, instantmesh_repo=None):

        self.base_dir = Path(base_dir)

        self.repo_dir = Path(
            instantmesh_repo or self.base_dir / "InstantMesh"
        )

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.zero123_pipeline = None
        self.controlnet_pipeline = None
        self.instantmesh_model = None

        self.snapshot_dir = (
            self.base_dir
            / "models/zero123plus/models--sudo-ai--zero123plus-v1.2"
            / "snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd"
        )

        self.zero123_unet_ckpt = (
            self.base_dir
            / "models/instantmesh/diffusion_pytorch_model.bin"
        )

        self.instantmesh_ckpt = (
            self.base_dir
            / "models/instantmesh/instant_mesh_large.ckpt"
        )

        self.controlnet_id = "lllyasviel/sd-controlnet-scribble"
        self.sd_model_id = "runwayml/stable-diffusion-v1-5"

        self._validate_paths()

    def _validate_paths(self):

        required_paths = [
            self.snapshot_dir,
            self.zero123_unet_ckpt,
            self.instantmesh_ckpt
        ]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model file not found: {path}"
                )

    def clear_gpu_memory(self):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def unload_zero123(self):

        if self.zero123_pipeline is not None:
            del self.zero123_pipeline
            self.zero123_pipeline = None

        self.clear_gpu_memory()

    def unload_controlnet(self):

        if self.controlnet_pipeline is not None:
            del self.controlnet_pipeline
            self.controlnet_pipeline = None

        self.clear_gpu_memory()

    def unload_instantmesh(self):

        if self.instantmesh_model is not None:
            del self.instantmesh_model
            self.instantmesh_model = None

        self.clear_gpu_memory()

    def load_zero123(self):

        if self.zero123_pipeline is not None:
            return self.zero123_pipeline

        if self.controlnet_pipeline is not None:
            self.unload_controlnet()

        self.zero123_pipeline = DiffusionPipeline.from_pretrained(
            str(self.snapshot_dir),
            custom_pipeline=str(self.repo_dir / "zero123plus"),
            torch_dtype=self.dtype,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Load InstantMesh-specific UNet weights
        state_dict = torch.load(
            str(self.zero123_unet_ckpt),
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

        self.zero123_pipeline.unet.load_state_dict(
            state_dict,
            strict=True
        )

        del state_dict
        self.clear_gpu_memory()

        self.zero123_pipeline.scheduler = (
            EulerAncestralDiscreteScheduler.from_config(
                self.zero123_pipeline.scheduler.config,
                timestep_spacing="trailing"
            )
        )

        self.zero123_pipeline = self.zero123_pipeline.to(
            device=self.device,
            dtype=self.dtype
        )

        return self.zero123_pipeline

    def load_controlnet(self):

        if self.controlnet_pipeline is not None:
            return self.controlnet_pipeline

        if self.zero123_pipeline is not None:
            self.unload_zero123()

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.dtype
        )

        self.controlnet_pipeline = (
            StableDiffusionControlNetPipeline.from_pretrained(
                self.sd_model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
        )

        self.controlnet_pipeline.enable_attention_slicing()

        self.controlnet_pipeline = (
            self.controlnet_pipeline.to(self.device)
        )

        return self.controlnet_pipeline

    def load_instantmesh(self, grid_res=80):

        if self.instantmesh_model is not None:
            return self.instantmesh_model

        if self.zero123_pipeline is not None:
            self.unload_zero123()

        if self.controlnet_pipeline is not None:
            self.unload_controlnet()

        # Apply Transformers compatibility patches
        import transformers.pytorch_utils as pytorch_utils

        def find_pruneable_heads_and_indices(
            heads,
            n_heads,
            head_size,
            already_pruned_heads
        ):
            mask = torch.ones(n_heads, head_size)

            heads = set(heads) - already_pruned_heads

            for head in heads:
                head -= sum(
                    1 for h in already_pruned_heads if h < head
                )
                mask[head] = 0

            index = torch.arange(mask.numel())[mask.view(-1).bool()]

            return heads, index.long()

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )

        from src.models.encoder.dino import ViTModel

        def compatible_get_head_mask(
            self,
            head_mask,
            num_hidden_layers
        ):
            if head_mask is None:
                return [None] * num_hidden_layers

            return self._convert_head_mask_to_5d(
                head_mask,
                num_hidden_layers
            )

        ViTModel.get_head_mask = compatible_get_head_mask

        # Load configuration
        config_path = (
            self.repo_dir / "configs/instant-mesh-large.yaml"
        )

        config = OmegaConf.load(str(config_path))
        model_config = config.model_config

        model_config.params.grid_res = grid_res

        self.instantmesh_model = instantiate_from_config(
            model_config
        )

        # Load checkpoint
        checkpoint = torch.load(
            str(self.instantmesh_ckpt),
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

        state_dict = checkpoint.get(
            "state_dict",
            checkpoint
        )

        state_dict = {
            key.replace("lrm_generator.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("lrm_generator.")
        }

        self.instantmesh_model.load_state_dict(
            state_dict,
            strict=True
        )

        del checkpoint, state_dict
        self.clear_gpu_memory()

        self.instantmesh_model = (
            self.instantmesh_model.to(self.device)
        )

        self.instantmesh_model.eval()

        self.instantmesh_model.init_flexicubes_geometry(
            device=self.device,
            fovy=50.0
        )

        return self.instantmesh_model
