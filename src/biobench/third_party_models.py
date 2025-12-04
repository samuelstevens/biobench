import logging
import os

import beartype
import einops
import torch
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

    def __init__(self, ckpt: str, drop_keys: list[str] | None = None, **_):
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
        elif ckpt.startswith("local:"):
            # Format: "local:ARCH/PATH_TO_CHECKPOINT"
            ckpt = ckpt.removeprefix("local:")
            arch, local_path = ckpt.split("/", 1)
            clip, self.img_transform = self._load_clip_skeleton(arch)
            state_dict = self._patch_state_dict(local_path, drop_keys or [])
            clip.load_state_dict(state_dict, strict=True)
        else:
            arch, ckpt = ckpt.split("/")
            clip, self.img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        self.model = clip.visual
        self.model.output_tokens = True  # type: ignore

    @staticmethod
    def _load_clip_skeleton(arch: str):
        import open_clip

        # returns (model, eval_transform)
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=None
        )
        return model, preprocess

    @staticmethod
    def _patch_state_dict(path: str, drop_keys: list[str]) -> dict:
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        # open_clip stores state_dict, optimizer, etc. We want the state_dict.
        state_dict = state_dict.get("state_dict", state_dict)
        # Often a DDP-'module.' prefix.
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k in drop_keys:
            state_dict.pop(k, None)
        return state_dict

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

    This class provides an interface to use any model from the Timm library as a vision backbone in the biobench framework.
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
        bsz_orig, _, _, _ = batch.shape
        feats = self.model.forward_features(batch)
        if feats.ndim == 4:
            # This could be a convnet of some kind with (batch, dim, width, height) or a ViT with (batch, width, height, dim).
            # We do some shape checking to figure it out.
            bsz, d1, d2, d3 = feats.shape

            if bsz_orig != bsz:
                msg = f"Batch size changed from {bsz_orig} to {bsz} in {self.ckpt}.forward_features()"
                raise ValueError(msg)

            if d1 == d2 and d3 > d1 and d3 > d2:
                bsz, w, h, d = feats.shape
                patches = einops.rearrange(feats, "b w h d -> b (w h) d")
            elif d2 == d3 and d1 > d2 and d1 > d3:
                bsz, d, w, h = feats.shape
                patches = einops.rearrange(feats, "b d w h -> b (w h) d")
            else:
                msg = f"Can't interpret shape {feats.shape} for model '{self.ckpt}'."
                raise ValueError(msg)

            # Validate the shape of the features
            if not (d > w and d > h):
                msg = f"Expected feature dimensions (d={d}) to be larger than spatial dimensions (w={w}, h={h}). This suggests the tensor dimensions may be in an unexpected order."
                raise ValueError(msg)

            if w != h:
                msg = f"Expected equal spatial dimensions, but got width={w} and height={h}. Unequal spatial dimensions indicate a mistake in our understanding of the output features from model '{self.ckpt}'."
                raise ValueError(msg)

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

        self.model = torch.hub.load("facebookresearch/dinov2", ckpt)

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        dct = self.model.forward_features(batch)

        return registry.EncodedImgBatch(
            dct["x_norm_clstoken"], dct["x_norm_patchtokens"]
        )

    def make_img_transform(self):
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])


@jaxtyped(typechecker=beartype.beartype)
class SAM2(registry.VisionBackbone):
    """
    A very small wrapper around the SAM-2 Hiera backbones exposed by `timm`.

    Design choices:

    * We rely 100 % on `timm` for model construction & weight loading (`timm.create_model("hf_hub:timm/<model_name>", pretrained=True)`).
    * The image transform is the exact one timm used during pre-training: `data.create_transform(**resolve_data_config(model.pretrained_cfg))`.
    """

    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import timm

        self.ckpt = ckpt

        self.model = timm.create_model(f"hf_hub:timm/{ckpt}", pretrained=True)

        self.data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)

    def make_img_transform(self):
        import timm

        return timm.data.create_transform(**self.data_cfg)

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        x = self.model.forward_features(batch)
        x = einops.rearrange(x, "b w h d -> b (w h) d")
        return registry.EncodedImgBatch(x.max(dim=1).values, x)
