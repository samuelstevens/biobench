Module biobench.tasks.imagenet
==============================
# Linear classification with [ImageNet 2012](https://image-net.org/)

This script uses the [Huggingface Hub ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) which needs permission to access.
Because `Args` is fully documented, running this script with `--help` will document all the configuration options.

## Ongoing Thoughts:

Prior work ([Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)) describes their linear probing as fitting a closed-form solution to a linear regression problem.
However, this [Cross Validated post](https://stats.stackexchange.com/questions/430341/linear-regression-for-multi-class-classification) suggests that minimizing least-squared error is not a reasonable approach.
I'm not sure which is better: closed-form solutions ("correct") or iterative optimization (easy, simple).

Functions
---------

`build_dataloader(args: biobench.tasks.imagenet.Args, transform, *, train: bool)`
:   Constructs a dataloader from `Args`, a transform from `src.biology_benchmark.models.interfaces.VisionModel.make_img_transform` and a boolean indicating whether we are in train mode.

`main(args: biobench.tasks.imagenet.Args)`
:   

`to_aim_value(value: object)`
:   Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.
    
    As a fallback, tries to call `to_aim_value()` on an object.

Classes
-------

`Args(seed: int = 42, n_epochs: int = 8, model: biobench.models.Params = <factory>, dtype: str = 'float16', learning_rate: float = 0.0003, weight_decay: float = 0.1, shuffle_buffer_size: int = 1000, batch_size: int = 2048, n_workers: int = 4, log_every: int = 1, device: str = 'cpu')`
:   Args(seed: int = 42, n_epochs: int = 8, model: biobench.models.Params = <factory>, dtype: str = 'float16', learning_rate: float = 0.0003, weight_decay: float = 0.1, shuffle_buffer_size: int = 1000, batch_size: int = 2048, n_workers: int = 4, log_every: int = 1, device: str = 'cpu')

    ### Class variables

    `batch_size: int`
    :   batch size for linear model.

    `device: str`
    :   (computed by program) which device to use (CUDA, cpu, mips, etc).

    `dtype: str`
    :   dtype to use for the model's forward pass.

    `learning_rate: float`
    :   learning rate for linear model.

    `log_every: int`
    :   how often to log to aim.

    `model: biobench.models.Params`
    :

    `n_epochs: int`
    :   number of training epochs.

    `n_workers: int`
    :   number of dataloader worker processes.

    `seed: int`
    :   random seed.

    `shuffle_buffer_size: int`
    :   how many samples to download before shuffling.

    `weight_decay: float`
    :   weight decay for linear model.

`LinearClassifier()`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, features: jaxtyping.Float[Tensor, 'batch dim']) ‑> jaxtyping.Float[Tensor, 'batch 1_000']`
    :