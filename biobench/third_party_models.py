import logging
import os

import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from biobench import interfaces

logger = logging.getLogger("third_party")


@beartype.beartype
def get_cache_dir() -> str:
    cache_dir = ""
    for var in ("BIOBENCH_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


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
class OpenClip(interfaces.VisionBackbone):
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
                arch, pretrained=ckpt, cache_dir=get_cache_dir()
            )

        self.model = clip.visual
        self.model.output_tokens = True  # type: ignore

    def make_img_transform(self):
        return self.img_transform

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> interfaces.EncodedImgBatch:
        result = self.model(batch)
        # Sometimes the model does not return patch features if it has none.
        if isinstance(result, tuple):
            img, patches = result
            return interfaces.EncodedImgBatch(img, patches)
        else:
            return interfaces.EncodedImgBatch(result, None)


@jaxtyped(typechecker=beartype.beartype)
class TimmVit(interfaces.VisionBackbone):
    """ """

    # TODO: docs + describe the ckpt format.
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import timm

        err_msg = "You are trying to load a non-ViT checkpoint; the `img_encode()` method assumes `model.forward_features()` will return features with shape (batch, n_patches, dim) which is not true for non-ViT checkpoints."
        assert "vit" in ckpt, err_msg
        self.model = timm.create_model(ckpt, pretrained=True)

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.img_transform = timm.data.create_transform(**data_cfg)

    def make_img_transform(self):
        return self.img_transform

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> interfaces.EncodedImgBatch:
        patches = self.model.forward_features(batch)
        # Use [CLS] token if it exists, otherwise do a maxpool
        if self.model.num_prefix_tokens > 0:
            img = patches[:, 0, ...]
        else:
            img = patches.max(axis=1).values

        # Remove all non-image patches, like the [CLS] token or registers
        patches = patches[:, self.model.num_prefix_tokens :, ...]

        return interfaces.EncodedImgBatch(img, patches)


@jaxtyped(typechecker=beartype.beartype)
class TorchvisionModel(interfaces.VisionBackbone):
    def __init__(self, ckpt: str):
        import torchvision

        arch, weights = ckpt.split("/")
        self.model = getattr(torchvision, arch)(weights=weights)
        self.model.eval()

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> interfaces.EncodedImgBatch:
        breakpoint()

    def make_img_transform(self):
        # Per the docs, each set of weights has its own transform: https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
        return self.model.weights.transforms()
