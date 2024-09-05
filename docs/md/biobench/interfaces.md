Module biobench.interfaces
==========================

Functions
---------

`get_git_hash()`
:   

`load_vision_backbone(args: biobench.interfaces.VisionBackboneArgs) ‑> biobench.interfaces.VisionBackbone`
:   

Classes
-------

`BenchmarkReport(name: str, score: float, argv: list[str] = <factory>, commit: str = '65e61be773df264727c70c65bdc69fd342fe4253', posix_time: float = <factory>, gpu_name: str = <factory>, hostname: str = <factory>)`
:   The result of running a benchmark.
    
    TODO: this needs to store more than just a summary statistic (`score`). It should include many raw results that can be used for analysis later on. It can even reference invidividual examples in a dataset so that they can be viewed.
    
    This should probably be in the form of
    
    summary: float
    splits: dict[str, float]
    examples: list[tuple[object, float, dict[str, object]]]
    
    See notebooks/tutorial.py for details.

    ### Class variables

    `argv: list[str]`
    :   command used to get this report.

    `commit: str`
    :   Git commit for this current report.

    `gpu_name: str`
    :

    `hostname: str`
    :

    `name: str`
    :   the benchmark name.

    `posix_time: float`
    :   time when this report was constructed.

    `score: float`
    :   mean score across the entire task, as a number between 0 and 1.

`EncodedImgBatch(img_features: jaxtyping.Float[Tensor, 'batch img_dim'], patch_features: jaxtyping.Float[Tensor, 'batch n_patches patch_dim'] | None)`
:   The output of a `VisionBackbone`'s `img_encode()` method.

    ### Class variables

    `img_features: jaxtyping.Float[Tensor, 'batch img_dim']`
    :   Image-level features. Each image is represented by a single vector.

    `patch_features: jaxtyping.Float[Tensor, 'batch n_patches patch_dim'] | None`
    :   Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such.

`VisionBackbone(*args, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * biobench.third_party_models.OpenClip
    * biobench.third_party_models.TimmViT

    ### Methods

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> biobench.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :

`VisionBackboneArgs(org: Literal['open_clip', 'timm-vit'] = 'open_clip', ckpt: str = 'RN50/openai')`
:   VisionBackboneArgs(org: Literal['open_clip', 'timm-vit'] = 'open_clip', ckpt: str = 'RN50/openai')

    ### Class variables

    `ckpt: str`
    :   The org-specific string. Will error if you pass the wrong one.

    `org: Literal['open_clip', 'timm-vit']`
    :   Where to load models from.