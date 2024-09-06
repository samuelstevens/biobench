Module biobench.interfaces
==========================

Functions
---------

`get_git_hash()`
:   

Classes
-------

`BenchmarkReport(name: str, examples: list[tuple[str, float, dict[str, object]]], splits: dict[str, float], argv: list[str] = <factory>, commit: str = '55bd3eb39223801f3b2d59382ceaa0335f7d119e', posix_time: float = <factory>, gpu_name: str = <factory>, hostname: str = <factory>)`
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

    `examples: list[tuple[str, float, dict[str, object]]]`
    :   a list of (example_id, score, info) tuples

    `gpu_name: str`
    :

    `hostname: str`
    :

    `name: str`
    :   the benchmark name.

    `posix_time: float`
    :   time when this report was constructed.

    `splits: dict[str, float]`
    :   individual splits and scores; can be anything you want.

    ### Instance variables

    `mean_score: float`
    :   mean score across the entire task, as a number between 0 and 1.

    ### Methods

    `get_confidence_interval(self, statistic='mean', confidence: float = 95, n_resamples: int = 500, seed: int = 42) ‑> tuple[float, float]`
    :   confidence interval for the statistics (mean) by bootstrapping individual scores of the examples.

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
    * biobench.third_party_models.TimmVit

    ### Methods

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> biobench.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :