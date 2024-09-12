import os

import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from biobench import interfaces


def get_cache_dir() -> str:
    cache_dir = ""
    for var in ("BIOBENCH_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


class OpenClip(interfaces.VisionBackbone):
    @jaxtyped(typechecker=beartype.beartype)
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        import open_clip

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

    @jaxtyped(typechecker=beartype.beartype)
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


class TimmVit(interfaces.VisionBackbone):
    @jaxtyped(typechecker=beartype.beartype)
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

    @jaxtyped(typechecker=beartype.beartype)
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
