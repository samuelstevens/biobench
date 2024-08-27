"""
# Linear classification with [ImageNet 2012](https://image-net.org/)

This script uses the [Huggingface Hub ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) which needs permission to access.
Because `Args` is fully documented, running this script with `--help` will document all the configuration options.

## Ongoing Thoughts:

Prior work ([Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)) describes their linear probing as fitting a closed-form solution to a linear regression problem.
However, this [Cross Validated post](https://stats.stackexchange.com/questions/430341/linear-regression-for-multi-class-classification) suggests that minimizing least-squared error is not a reasonable approach.
I'm not sure which is better: closed-form solutions ("correct") or iterative optimization (easy, simple).
"""

import dataclasses
import logging
import multiprocessing
import time
import warnings

import aim
import beartype
import datasets
import torch
import torchmetrics
import tyro
from jaxtyping import Float, jaxtyped
from torch import Tensor

from biology_benchmark import models

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@beartype.beartype
@dataclasses.dataclass
class Args:
    seed: int = 42
    """random seed."""
    n_epochs: int = 8
    """number of training epochs."""

    model: models.Params = dataclasses.field(default_factory=lambda: models.Params())
    dtype: str = "float16"
    """dtype to use for the model's forward pass."""
    learning_rate: float = 3e-4
    """learning rate for linear model."""
    weight_decay: float = 0.1
    """weight decay for linear model."""

    # Data
    shuffle_buffer_size: int = 1_000
    """how many samples to download before shuffling."""
    batch_size: int = 2048
    """batch size for linear model."""
    n_workers: int = 4
    """number of dataloader worker processes."""

    log_every: int = 1
    """how often to log to aim."""

    # Computed at runtime.
    device: str = "cpu"
    """(computed by program) which device to use (CUDA, cpu, mips, etc)."""


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.linear = torch.nn.LazyLinear(1_000)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, features: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch 1_000"]:
        logits = self.linear(features)
        return torch.nn.functional.softmax(logits, dim=-1)


def to_aim_value(value: object):
    """
    Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.

    As a fallback, tries to call `to_aim_value()` on an object.
    """
    if value is None:
        return value

    if isinstance(value, (str, int, float)):
        return value

    if isinstance(value, list):
        return [to_aim_value(elem) for elem in value]

    if isinstance(value, dict):
        return {to_aim_value(k): to_aim_value(v) for k, v in value.items()}

    if dataclasses.is_dataclass(value):
        return to_aim_value(dataclasses.asdict(value))

    try:
        return value.to_aim_value()
    except AttributeError as err:
        raise err


@beartype.beartype
def build_dataloader(args: Args, transform, *, train: bool):
    """
    Constructs a dataloader from `Args`, a transform from `src.biology_benchmark.models.interfaces.VisionModel.make_img_transform` and a boolean indicating whether we are in train mode.
    """
    split = "train" if train else "val"
    drop_last = train
    shuffle = train

    dataset = datasets.load_dataset("ILSVRC/imagenet-1k", streaming=True, split=split)
    if shuffle:
        dataset.shuffle(args.seed, buffer_size=args.shuffle_buffer_size)

    def hf_transform(example):
        example["image"] = transform(example["image"])
        return example

    dataset = dataset.map(hf_transform).with_format("torch")

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=drop_last,
        num_workers=min(args.n_workers, dataset.n_shards),
        # TODO: Evaluate whether this helps throughput or not on A100 and on A6000.
        pin_memory=True,
        persistent_workers=(args.n_workers > 0),
    )


def main(args: Args):
    # 0. Do some runtime-based modification of args.

    # 1. Load model.
    deep_model = models.load_model(args.model)
    # Freeze model
    for param in deep_model.parameters():
        param.requires_grad = False
    # Compile for speed.
    deep_model = torch.compile(deep_model)

    linear_model = LinearClassifier()

    # 2. Load dataloaders.
    transform = deep_model.make_img_transform()
    train_dataloader = build_dataloader(args, transform, train=True)
    # val_dataloader = build_dataloader(args, transform, train=False)

    # 3. Load optimizers.
    ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, args.dtype))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        linear_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # 4. Load tracking.
    logger = logging.getLogger("imagnet")
    task = "multiclass"
    train_metrics = {
        "train_acc1": torchmetrics.Accuracy(task=task, top_k=1, num_classes=1_000),
        "train_acc5": torchmetrics.Accuracy(task=task, top_k=5, num_classes=1_000),
    }
    train_metrics = torchmetrics.MetricCollection(train_metrics).to(args.device)
    run = aim.Run()
    hparams = {key: to_aim_value(value) for key, value in vars(args).items()}
    run["hparams"] = hparams

    global_step = 0
    global_start_s = time.time()
    with warnings.catch_warnings():
        # .with_format("torch") will throw a userwarning. I could get rid of the with_format("torch") call and it would still return torch tensors, but I would prefer to simply ignore the warnings for now.
        warnings.simplefilter("ignore", UserWarning)

        for epoch in range(args.n_epochs):
            # Train
            for batch, examples in enumerate(train_dataloader):
                with ctx:
                    encoded = deep_model.img_encode(examples["image"])
                outputs = linear_model(encoded.img_features)

                loss = loss_fn(outputs, examples["label"])
                loss.backward()

                # Step optimizer
                optimizer.step()
                # This has to be after optimizer.step() or scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_metrics.reset()
                train_metrics(outputs, examples["label"])

                if batch % (args.log_every) == 0:
                    imgs_per_sec = (
                        global_step * args.batch_size / (time.time() - global_start_s)
                    )
                    # We can use loss.item() because we don't need to sync it across all processes since it's going to be noisy anyways.
                    metrics = {
                        **train_metrics.compute(),
                        "train_cross_entropy": loss.item(),
                        "train_step": global_step,
                        "perf_images_per_sec": imgs_per_sec,
                    }
                    run.track(metrics, step=global_step)
                    logger.info(
                        "step: %s, loss: %.4g, imgs/sec: %.2g",
                        global_step,
                        loss.item(),
                        imgs_per_sec,
                    )

            # Validate
            breakpoint()


if __name__ == "__main__":
    # Required on macOS so we don't use pickling to move things between processes.
    multiprocessing.set_start_method("fork")
    main(tyro.cli(Args))
