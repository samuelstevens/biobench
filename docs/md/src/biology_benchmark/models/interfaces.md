Module src.biology_benchmark.models.interfaces
==============================================

Classes
-------

`EncodedImgBatch(img_features: jaxtyping.Float[Tensor, 'batch dim'], patch_features: jaxtyping.Float[Tensor, 'batch patch_size dim'] | None)`
:   EncodedImgBatch(img_features: jaxtyping.Float[Tensor, 'batch dim'], patch_features: jaxtyping.Float[Tensor, 'batch patch_size dim'] | None)

    ### Class variables

    `img_features: jaxtyping.Float[Tensor, 'batch dim']`
    :

    `patch_features: jaxtyping.Float[Tensor, 'batch patch_size dim'] | None`
    :   Patch-level features. Only ViTs have patch-level features.

`VisionModel(*args, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * src.biology_benchmark.models.third_party.OpenClip

    ### Methods

    `get_image_size(self) ‑> tuple[int, int]`
    :   Returns (width, height) as a tuple of ints

    `get_patch_size(self) ‑> int`
    :   Returns an int

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> src.biology_benchmark.models.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :