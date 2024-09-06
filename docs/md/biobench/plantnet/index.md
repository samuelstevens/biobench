Module biobench.plantnet
========================

Sub-modules
-----------
* biobench.plantnet.download

Functions
---------

`benchmark(backbone: biobench.interfaces.VisionBackbone, args: biobench.plantnet.Args) ‑> biobench.interfaces.BenchmarkReport`
:   Steps:
    1. Get features for all images.
    2. Select lambda using validation data.
    3. Report score on test data.

Classes
-------

`Args(dataset_dir: str = '', batch_size: int = 256, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cuda')`
:   Args(dataset_dir: str = '', batch_size: int = 256, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cuda')

    ### Class variables

    `batch_size: int`
    :   batch size for deep model.

    `dataset_dir: str`
    :   dataset directory; where you downloaded Pl@ntNet to. It should contain plantnet300K_metadata.json and images/

    `device: Literal['cpu', 'cuda']`
    :   (computed at runtime) which kind of accelerator to use.

    `n_workers: int`
    :   number of dataloader worker processes.