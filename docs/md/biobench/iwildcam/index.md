Module biobench.iwildcam
========================
Fits a linear classifier that is trained using cross-entropy on the training set of iWildCam 2020.

Sub-modules
-----------
* biobench.iwildcam.download

Functions
---------

`benchmark(args: biobench.iwildcam.Args, model_args: tuple[str, str]) ‑> tuple[tuple[str, str], biobench.interfaces.TaskReport]`
:   

`get_features(args: biobench.iwildcam.Args, backbone: biobench.interfaces.VisionBackbone, dataloader) ‑> biobench.iwildcam.Features`
:   

`init_ridge(args: biobench.iwildcam.Args)`
:   

Classes
-------

`Args(seed: int = 42, datadir: str = '', device: str = 'cuda', debug: bool = False, batch_size: int = 2048, n_workers: int = 4)`
:   Args(seed: int = 42, datadir: str = '', device: str = 'cuda', debug: bool = False, batch_size: int = 2048, n_workers: int = 4)

    ### Ancestors (in MRO)

    * biobench.interfaces.TaskArgs

    ### Class variables

    `batch_size: int`
    :   batch size for deep model.

    `n_workers: int`
    :   number of dataloader worker processes.

`Features(x: jaxtyping.Float[ndarray, 'n dim'], y: jaxtyping.Int[ndarray, 'n n_classes'], ids: jaxtyping.Shaped[ndarray, 'n'])`
:   Features(x: jaxtyping.Float[ndarray, 'n dim'], y: jaxtyping.Int[ndarray, 'n n_classes'], ids: jaxtyping.Shaped[ndarray, 'n'])

    ### Class variables

    `ids: jaxtyping.Shaped[ndarray, 'n']`
    :

    `x: jaxtyping.Float[ndarray, 'n dim']`
    :

    `y: jaxtyping.Int[ndarray, 'n n_classes']`
    :

`MeanScoreCalculator()`
: