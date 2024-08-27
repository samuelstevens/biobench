import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import interfaces


class OpenClip(interfaces.VisionModel):
    @jaxtyped(typechecker=beartype.beartype)
    def __init__(self, arch: str, ckpt: str, **kwargs):
        super().__init__()
        import open_clip

        clip, self.img_transform = open_clip.create_model_from_pretrained(  # type: ignore
            arch, pretrained=ckpt
        )

        self.model = clip.visual
        self.model.output_tokens = True  # type: ignore

        # Set patch_size and image_size
        config = open_clip.get_model_config(arch)
        self.patch_size = config["vision_cfg"]["patch_size"]  # type: ignore

        size = config["vision_cfg"]["image_size"]  # type: ignore
        self.image_size = (size, size)

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

    @beartype.beartype
    @staticmethod
    def parse_model_str(model: str) -> tuple[str, str]:
        """Parse a string like 'RN50/openai' into 'RN50', 'openai' for use with the open_clip package."""
        arch, ckpt = model.split("/")
        return arch, ckpt
