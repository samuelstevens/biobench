Module biobench.interfaces
==========================

Classes
-------

`BenchmarkReport(name: str, score: float)`
:   The result of running a benchmark.

    ### Class variables

    `name: str`
    :   the benchmark name.

    `score: float`
    :   mean score across the entire task, as a number between 0 and 1.

`EncodedImgBatch(img_features: jaxtyping.Float[Tensor, 'batch img_dim'], patch_features: jaxtyping.Float[Tensor, 'batch patch_size patch_dim'] | None)`
:   The output of a `VisionBackbone`'s `img_encode()` method.

    ### Class variables

    `img_features: jaxtyping.Float[Tensor, 'batch img_dim']`
    :   Image-level features. Each image is represented by a single vector.

    `patch_features: jaxtyping.Float[Tensor, 'batch patch_size patch_dim'] | None`
    :   Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such.

`VisionBackbone(*args, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * biobench.third_party_models.OpenClip

    ### Methods

    `get_image_size(self) ‑> tuple[int, int]`
    :   Returns (width, height) as a tuple of ints

    `get_patch_size(self) ‑> int`
    :   Returns an int

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> biobench.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :