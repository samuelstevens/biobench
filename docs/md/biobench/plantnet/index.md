Module biobench.plantnet
========================

Sub-modules
-----------
* biobench.plantnet.download

Functions
---------

`benchmark(args: biobench.plantnet.Args, model_args: tuple[str, str]) ‑> tuple[tuple[str, str], biobench.interfaces.TaskReport]`
:   Steps:
    1. Get features for all images.
    2. Select lambda using validation data.
    3. Report score on test data.

Classes
-------

`Args(seed: int = 42, datadir: str = '', device: str = 'cuda', debug: bool = False, batch_size: int = 256, n_workers: int = 4)`
:   Args(seed: int = 42, datadir: str = '', device: str = 'cuda', debug: bool = False, batch_size: int = 256, n_workers: int = 4)

    ### Ancestors (in MRO)

    * biobench.interfaces.TaskArgs

    ### Class variables

    `batch_size: int`
    :   batch size for deep model.

    `n_workers: int`
    :   number of dataloader worker processes.