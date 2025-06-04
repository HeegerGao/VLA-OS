"""
base_vision.py

Abstract class definition of a Vision Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.

We also define the generic TimmViTBackbone class here, providing a default interface for loading any TIMM Vision
Transformer model for feature extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, List

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper

# === Utility Function for handling image sequences ===
def compute_sequence_patches(pixel_values: torch.Tensor, featurizer, seq_len: int):
    patches = {}
    # must be of shape B x T x C x H x W
    assert len(pixel_values.shape) == 5, "Pixel values must be (B, T, C, H, W)"
    assert pixel_values.shape[1] >= seq_len, "Sequence length was too short!"
    trunc_pixels = pixel_values[:, :seq_len]

    # will output B x (T*featurizer.num_patches) x embed_dim
    B, T, C, H, W = trunc_pixels.shape
    patches = featurizer(trunc_pixels.reshape(-1, C, H, W))[0].reshape(B, T, -1, featurizer.embed_dim).reshape(B, -1, featurizer.embed_dim)

    return patches

# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Special Image Transform Wrapper for sequences ===
class WrapSequenceImageTransform(ImageTransform):
    """Performs transforms and returns a tensor that with a sequence len dimension"""

    def __init__(self, base_transform: ImageTransform, sequence_len: int):
        self.base_transform = base_transform
        self.sequence_len = sequence_len
        assert self.sequence_len > 1

    def __call__(self, img: List[Image], **kwargs: str):
        assert isinstance(img, list) and len(img) == self.sequence_len, f"Input must be a list of images with length == {self.sequence_len}"
        outputs = [self.base_transform(image, **kwargs) for image in img]

        # return either a dict w/ stacked tensors along dim=0, or a single tensor
        # output tensors are (SEQ_LEN, C, H, W)
        if isinstance(outputs[0], dict):
            return {k: torch.stack([o[k] for o in outputs], dim=0) for k in outputs[0].keys()}
        elif isinstance(outputs[0], torch.Tensor):
            return torch.stack(outputs, dim=0)

# === Custom Torchvision Image Transforms ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


# === Abstract Base Class for arbitrary Vision Backbones ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, image_sequence_len = 1) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size
        self.image_sequence_len = image_sequence_len

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        if self.image_sequence_len == 1:
            return self.image_transform
        else:
            return WrapSequenceImageTransform(self.image_transform, sequence_len=self.image_sequence_len)

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


# === Abstract Base Class for Arbitrary TIMM Vision Transformer Backbones ===
class TimmViTBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        image_sequence_len: int = 1,
        override_act_layer: Optional[str] = None,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size, image_sequence_len=image_sequence_len)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        if self.override_act_layer is None:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
            )
        else:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size,
                act_layer=self.override_act_layer,
            )
        self.featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        if self.image_sequence_len == 1:
            return self.featurizer(pixel_values)
        else:
            return compute_sequence_patches(pixel_values, self.featurizer, self.image_sequence_len)
        
    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches  # TODO: check if we need to * self.image_sequence_len

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
