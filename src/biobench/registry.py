"""
Stores all vision backbones.
Users can register new custom backbones from their code to evaluate on biobench using `register_vision_backbone`.
As long as it satisfies the `VisionBackbone` interface, it will work will all tasks.
"""

import dataclasses
import logging

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import config

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedImgBatch:
    """The output of a `VisionBackbone`'s `VisionBackbone.img_encode()` method."""

    img_features: Float[Tensor, "batch img_dim"]
    """Image-level features. Each image is represented by a single vector."""
    patch_features: Float[Tensor, "batch n_patches patch_dim"] | None
    """Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such."""


@jaxtyped(typechecker=beartype.beartype)
class VisionBackbone(torch.nn.Module):
    """
    A frozen vision model that embeds batches of images into batches of vectors.

    To add new models to the benchmark, you can simply create a new class that satisfies this interface and register it.
    See `biobench.registry` for a tutorial on adding new vision backbones.
    """

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        """Encode a batch of images."""
        err_msg = f"{self.__class__.__name__} must implemented img_encode()."
        raise NotImplementedError(err_msg)

    def make_img_transform(self):
        """
        Return whatever function the backbone wants for image preprocessing.
        This should be an evaluation transform, not a training transform, because we are using the output features of this backbone as data and not updating this backbone.
        """
        err_msg = f"{self.__class__.__name__} must implemented make_img_transform()."
        raise NotImplementedError(err_msg)

    @classmethod
    @beartype.beartype
    def normalize_model_ckpt(cls, ckpt: str) -> str:
        """Normalize a checkpoint identifier to a canonical form. By default, returns the checkpoint unchanged. Models with machine-specific paths can override this to extract a canonical identifier."""
        return ckpt


_global_backbone_registry: dict[str, type[VisionBackbone]] = {}


@beartype.beartype
def load_vision_backbone(model_cfg: config.Model) -> VisionBackbone:
    """
    Load a pretrained vision backbone.
    """
    if model_cfg.org not in _global_backbone_registry:
        raise ValueError(f"Org '{model_cfg.org}' not found.")

    cls = _global_backbone_registry[model_cfg.org]
    return cls(model_cfg.ckpt, drop_keys=model_cfg.drop_keys)


@beartype.beartype
def register_vision_backbone(model_org: str, cls: type[VisionBackbone]):
    """
    Register a new vision backbone class.
    """
    if model_org in _global_backbone_registry:
        logger.warning("Overwriting key '%s' in registry.", model_org)
    _global_backbone_registry[model_org] = cls


def list_vision_backbones() -> list[str]:
    """
    List all vision backbone model orgs.
    """
    return list(_global_backbone_registry.keys())


@beartype.beartype
def normalize_model_ckpt(model_org: str, ckpt: str) -> str:
    """
    Normalize a checkpoint identifier to its canonical form using the backbone class's normalization method.
    """
    if model_org not in _global_backbone_registry:
        raise ValueError(f"Org '{model_org}' not found.")

    cls = _global_backbone_registry[model_org]
    return cls.normalize_model_ckpt(ckpt)
