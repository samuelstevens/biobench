Module biobench.interfaces
==========================

Functions
---------

`get_git_hash()`
:   

Classes
-------

`EncodedImgBatch(img_features: jaxtyping.Float[Tensor, 'batch img_dim'], patch_features: jaxtyping.Float[Tensor, 'batch n_patches patch_dim'] | None)`
:   The output of a `VisionBackbone`'s `img_encode()` method.

    ### Class variables

    `img_features: jaxtyping.Float[Tensor, 'batch img_dim']`
    :   Image-level features. Each image is represented by a single vector.

    `patch_features: jaxtyping.Float[Tensor, 'batch n_patches patch_dim'] | None`
    :   Patch-level features. Only ViTs have patch-level features. These features might be a different dimension that the image features because of projection heads or such.

`Example(id: str, score: float, info: dict[str, object])`
:   Example(id: str, score: float, info: dict[str, object])

    ### Class variables

    `id: str`
    :

    `info: dict[str, object]`
    :

    `score: float`
    :

`TaskArgs(seed: int = 42, datadir: str = '', device: str = 'cuda', debug: bool = False)`
:   Common args for all tasks.

    ### Descendants

    * biobench.iwildcam.Args
    * biobench.kabr.Args
    * biobench.newt.Args
    * biobench.plantnet.Args

    ### Class variables

    `datadir: str`
    :   dataset directory; where you downloaded this task's data to.

    `debug: bool`
    :   (computed at runtime) whether to run in debug mode.

    `device: str`
    :   (computed at runtime) which kind of accelerator to use.

    `seed: int`
    :   random seed.

`TaskReport(name: str, examples: list[biobench.interfaces.Example], splits: dict[str, float], calc_mean_score: Callable[[list[biobench.interfaces.Example]], float], argv: list[str] = <factory>, commit: str = '72249355f2b32fb866bd164a29e0161500b4a1a3', posix_time: float = <factory>, gpu_name: str = <factory>, hostname: str = <factory>)`
:   The result of running a benchmark task.
    
    See notebooks/tutorial.py for details.

    ### Class variables

    `argv: list[str]`
    :   command used to get this report.

    `calc_mean_score: Callable[[list[biobench.interfaces.Example]], float]`
    :   way to calculate mean score from a list of examples.

    `commit: str`
    :   Git commit for this current report.

    `examples: list[biobench.interfaces.Example]`
    :   a list of (example_id, score, info) objects

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

    ### Methods

    `get_confidence_interval(self, statistic='mean', confidence: float = 95, n_resamples: int = 500, seed: int = 42) ‑> tuple[float, float]`
    :   confidence interval for the statistics (mean) by bootstrapping individual scores of the examples.
        
        NOTE: it's crazy how much easier this would be in jax to vmap. PyTrees of Examples would simply contains batch dimensions, and then I would jax.vmap(get_mean_score)(batched_examples).

    `get_mean_score(self) ‑> float`
    :

    `to_dict(self) ‑> dict[str, object]`
    :

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