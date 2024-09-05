import dataclasses
import socket
import subprocess
import sys
import time
import typing

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
    patch_features: Float[Tensor, "batch n_patches patch_dim"] | None
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


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


Org = typing.Literal["open_clip", "timm-vit"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class VisionBackboneArgs:
    org: Org = "open_clip"
    """Where to load models from."""
    ckpt: str = "RN50/openai"
    """The org-specific string. Will error if you pass the wrong one."""


@beartype.beartype
def load_vision_backbone(args: VisionBackboneArgs) -> VisionBackbone:
    if args.org == "open_clip":
        import third_party_models

        arch, ckpt = third_party_models.OpenClip.parse_model_str(args.ckpt)
        return third_party_models.OpenClip(arch, ckpt)
    elif args.org == "timm-vit":
        import third_party_models

        return third_party_models.TimmViT(args.ckpt)
    else:
        typing.assert_never(args.org)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """
    The result of running a benchmark.

    TODO: this needs to store more than just a summary statistic (`score`). It should include many raw results that can be used for analysis later on. It can even reference invidividual examples in a dataset so that they can be viewed.

    This should probably be in the form of

    summary: float
    splits: dict[str, float]
    examples: list[tuple[object, float, dict[str, object]]]

    See notebooks/tutorial.py for details.
    """

    # Actual details of the report
    name: str
    """the benchmark name."""
    score: float
    """mean score across the entire task, as a number between 0 and 1."""

    # Stuff for trying to reproduce this result. These are filled in by default.
    argv: list[str] = dataclasses.field(default_factory=lambda: sys.argv)
    """command used to get this report."""
    commit: str = get_git_hash()
    """Git commit for this current report."""
    posix_time: float = dataclasses.field(default_factory=time.time)
    """time when this report was constructed."""
    gpu_name: str = dataclasses.field(
        default_factory=lambda: torch.cuda.get_device_properties(0).name
    )
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
