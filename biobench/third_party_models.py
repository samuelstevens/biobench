import logging
import os

import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import helpers, registry

logger = logging.getLogger("third_party")


@beartype.beartype
def get_ssl() -> bool:
    """
    Checks whether BIOBENCH_DISABLE_SSL is present in the environment.

    We use environment variables rather than a boolean argument because

    1. This is only needed on some systems, like OSC.
    2. Every benchmark needs it in exactly the same way, so it would show up in every benchmark script as more "noise".
    3. It is not manipulated throughout the running of the program. It's a global variable that's set at the start of the jobs.

    But in general, we should not use environment variables to manage program state.

    Returns:
        A boolean that's true if we should use SSL and false if not.
    """
    disable = os.environ.get("BIOBENCH_DISABLE_SSL", None)
    return not disable


@jaxtyped(typechecker=beartype.beartype)
class OpenClip(registry.VisionBackbone):
    """
    Loads checkpoints from [open_clip](https://github.com/mlfoundations/open_clip), an open-source reproduction of the original [CLIP](https://arxiv.org/abs/2103.00020) paper.

    Checkpoints are in the format `<ARCH>/<CKPT>`.
    Look at the [results file](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv) for the pretrained models.
    For example, to load a ViT-B/16 train on Apple's Data Filtering Networks dataset, you would use `ViT-B-16/dfn2b`.
    """

    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import open_clip

        if not get_ssl():
            logger.warning("Ignoring SSL certs. Try not to do this!")
            # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
            # Ideally we don't have to disable SSL but we are only downloading weights.
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context

        if ckpt.startswith("hf-hub:"):
            clip, self.img_transform = open_clip.create_model_from_pretrained(ckpt)
        else:
            arch, ckpt = ckpt.split("/")
            clip, self.img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        self.model = clip.visual
        self.model.output_tokens = True  # type: ignore

    def make_img_transform(self):
        return self.img_transform

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        result = self.model(batch)
        # Sometimes the model does not return patch features if it has none.
        if isinstance(result, tuple):
            img, patches = result
            return registry.EncodedImgBatch(img, patches)
        else:
            return registry.EncodedImgBatch(result, None)


@jaxtyped(typechecker=beartype.beartype)
class Timm(registry.VisionBackbone):
    """
    Wrapper for models from the Timm (PyTorch Image Models) library.

    This class provides an interface to use any model from the Timm library
    as a vision backbone in the biobench framework.
    """

    # TODO: docs + describe the ckpt format.
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import timm

        self.ckpt = ckpt

        self.model = timm.create_model(ckpt, pretrained=True)

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.img_transform = timm.data.create_transform(**data_cfg)

    def make_img_transform(self):
        return self.img_transform

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        feats = self.model.forward_features(batch)
        if feats.ndim == 4:
            # This is probably a convnet of some kind, with (batch, dim, width, height)
            bsz, d, w, h = feats.shape

            # Validate the shape of the features
            if not (d > w and d > h):
                raise ValueError(
                    f"Expected feature dimensions (d={d}) to be larger than spatial dimensions (w={w}, h={h}). This suggests the tensor dimensions may be in an unexpected order."
                )

            if w != h:
                raise ValueError(
                    f"Expected equal spatial dimensions, but got width={w} and height={h}. Unequal spatial dimensions indicate a mistake in our understanding of the output features from model '{self.ckpt}'."
                )

            # Reshape to (batch, patches, dim) format
            patches = feats.permute(0, 2, 3, 1).reshape(bsz, w * h, d)
            # TODO: should we only use max pooling?
            img = patches.max(dim=1).values  # Global max pooling
        elif feats.ndim == 3:
            # This is probably a ViT with (batch, patches, dim)
            bsz, num_patches, d = feats.shape
            patches = feats

            # For ViT models, we typically use the class token ([CLS]) as the image representation
            # if it exists, otherwise we do mean pooling over patches
            if self.model.num_prefix_tokens > 0:
                img = patches[:, 0]  # Use [CLS] token
                # Remove prefix tokens (like [CLS]) from patches
                patches = patches[:, self.model.num_prefix_tokens :]
            else:
                img = patches.max(dim=1).values  # Max pooling if no [CLS] token
        else:
            raise ValueError(
                f"Unexpected feature dimension: {feats.ndim}. Expected either 3 (ViT models) or 4 (ConvNet models). Check if the model architecture {self.ckpt} is supported."
            )

        return registry.EncodedImgBatch(img, patches)


@jaxtyped(typechecker=beartype.beartype)
class DinoV2(registry.VisionBackbone):
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()

        import torch

        self.model = torch.hub.load("facebookresearch/dinov2", ckpt)

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        dct = self.model.forward_features(batch)

        return registry.EncodedImgBatch(
            dct["x_norm_clstoken"], dct["x_norm_patchtokens"]
        )

    def make_img_transform(self):
        import torch
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])


@jaxtyped(typechecker=beartype.beartype)
class TorchvisionModel(registry.VisionBackbone):
    def __init__(self, ckpt: str):
        import torchvision

        arch, weights = ckpt.split("/")
        self.model = getattr(torchvision, arch)(weights=weights)
        self.model.eval()

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        breakpoint()
