Module biobench.newt
====================
# NeWT: Natural World Tasks

NeWT is a collection of 164 binary classification tasks related to visual understanding of the natural world ([CVPR 2021 paper](https://arxiv.org/abs/2103.16483), [code](https://github.com/visipedia/newt/tree/main)).

We evaluate a vision model by extracting visual features for each image, fitting a linear SVM to the training examples, and evaluating on the test data.
We aggregate scores across all 164 tasks.

If you use this evaluation, be sure to cite the original work:

```
@inproceedings{van2021benchmarking,
  title={Benchmarking Representation Learning for Natural World Image Collections},
  author={Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
```

Sub-modules
-----------
* biobench.newt.download

Functions
---------

`benchmark(args: biobench.newt.Args, model_args: tuple[str, str]) ‑> tuple[tuple[str, str], biobench.interfaces.TaskReport]`
:   Steps:
    1. Get features for all images.
    2. Select subsets of the features for fitting with SVMs.
    3. Evaluate SVMs and report.

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