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

`benchmark(backbone: biobench.interfaces.VisionBackbone, args: biobench.newt.Args) ‑> biobench.interfaces.BenchmarkReport`
:   Steps:
    1. Get features for all images.
    2. Select subsets of the features for fitting with SVMs.
    3. Evaluate SVMs and report.

Classes
-------

`Args(dataset_dir: str = '', batch_size: int = 256, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cuda')`
:   Args(dataset_dir: str = '', batch_size: int = 256, n_workers: int = 4, device: Literal['cpu', 'cuda'] = 'cuda')

    ### Class variables

    `batch_size: int`
    :   batch size for deep model.

    `dataset_dir: str`
    :   dataset directory; where you downloaded NEWT to.

    `device: Literal['cpu', 'cuda']`
    :   (computed at runtime) which kind of accelerator to use.

    `n_workers: int`
    :   number of dataloader worker processes.