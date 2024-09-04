import dataclasses

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedImgBatch:
    """The output of a `VisionBackbone`'s `img_encode()` method."""

    img_features: Float[Tensor, "batch img_dim"]
    """Image-level features. Each image is represented by a single vector."""
    patch_features: Float[Tensor, "batch patch_size patch_dim"] | None
    """Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such."""


@jaxtyped(typechecker=beartype.beartype)
class VisionBackbone(torch.nn.Module):
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> EncodedImgBatch:
        err_msg = f"{self.__class__.__name__} must implemented img_encode()."
        raise NotImplementedError(err_msg)

    def make_img_transform(self):
        err_msg = f"{self.__class__.__name__} must implemented make_img_transform()."
        raise NotImplementedError(err_msg)

    @jaxtyped(typechecker=beartype.beartype)
    def get_patch_size(self) -> int:
        """Returns an int"""
        if hasattr(self, "patch_size") and isinstance(self.patch_size, int):
            return self.patch_size

        err_msg = f"{self.__class__.__name__} must implemented get_patch_size()."
        raise NotImplementedError(err_msg)

    @jaxtyped(typechecker=beartype.beartype)
    def get_image_size(self) -> tuple[int, int]:
        """Returns (width, height) as a tuple of ints"""
        if hasattr(self, "image_size") and isinstance(self.image_size, tuple):
            return self.image_size

        err_msg = f"{self.__class__.__name__} must implemented get_img_size()."
        raise NotImplementedError(err_msg)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """The result of running a benchmark."""

    name: str
    """the benchmark name."""

    score: float
    """mean score across the entire task, as a number between 0 and 1."""
