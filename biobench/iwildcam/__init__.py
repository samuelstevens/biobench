"""
Fits a linear classifier that is trained using cross-entropy on the training set of iWildCam 2020.


"""

import dataclasses
import logging
import time

import beartype
import torch
import torchmetrics
import tqdm
import wilds
import wilds.common.data_loaders
from jaxtyping import Float, jaxtyped
from torch import Tensor

from biobench import interfaces

logger = logging.getLogger("newt")


@beartype.beartype
@dataclasses.dataclass
class Args:
    seed: int = 42
    """random seed."""
    n_epochs: int = 8
    """number of training epochs."""
    dtype: str = "float16"
    """dtype to use for the model's forward pass."""
    learning_rate: float = 3e-4
    """learning rate for linear model."""
    weight_decay: float = 0.1
    """weight decay for linear model."""

    # Data
    dataset_dir: str = ""
    """dataset directory; where you downloaded iWildCam to."""
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
    def __init__(self, n_classes: int):
        super().__init__()
        self.linear = torch.nn.LazyLinear(n_classes)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, features: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch n_classes"]:
        logits = self.linear(features)
        return torch.nn.functional.softmax(logits, dim=-1)


@beartype.beartype
def benchmark(
    backbone: interfaces.VisionBackbone, args: Args
) -> interfaces.BenchmarkReport:
    # 1. Load dataloaders.
    transform = backbone.make_img_transform()
    dataset = wilds.get_dataset(
        dataset="iwildcam", download=False, root_dir=args.dataset_dir
    )
    train_dataset = dataset.get_subset("train", transform=transform)
    train_dataloader = wilds.common.data_loaders.get_train_loader(
        "standard",
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    test_data = dataset.get_subset("test", transform=transform)
    test_dataloader = wilds.common.data_loaders.get_eval_loader(
        "standard",
        test_data,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    # 2. Some modeling stuff
    # Freeze model
    for param in backbone.parameters():
        param.requires_grad = False
    # Compile for speed.
    backbone = torch.compile(backbone.to(args.device))
    linear_model = LinearClassifier(dataset.n_classes).to(args.device)

    # 3. Load optimizers.
    ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, args.dtype))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        linear_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # 4. Load tracking.
    train_acc1 = torchmetrics.Accuracy(
        task="multiclass", top_k=1, num_classes=dataset.n_classes
    )
    train_acc5 = torchmetrics.Accuracy(
        task="multiclass", top_k=5, num_classes=dataset.n_classes
    )
    train_metrics = {"train_acc1": train_acc1, "train_acc5": train_acc5}
    train_metrics = torchmetrics.MetricCollection(train_metrics).to(args.device)

    global_step = 0
    global_start_s = time.time()
    for epoch in range(args.n_epochs):
        # Train
        for batch, (images, labels, metadata) in enumerate(train_dataloader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            with ctx:
                encoded = backbone.img_encode(images)
                outputs = linear_model(encoded.img_features)

            loss = loss_fn(outputs, labels)
            loss.backward()

            # Step optimizer
            optimizer.step()
            # This has to be after optimizer.step() or scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            train_metrics.reset()
            train_metrics(outputs, labels)

            if batch % (args.log_every) == 0:
                imgs_per_sec = (
                    global_step * args.batch_size / (time.time() - global_start_s)
                )
                # We can use loss.item() because we don't need to sync it across all processes since it's going to be noisy anyways.
                # metrics = {
                #     **train_metrics.compute(),
                #     "train_cross_entropy": loss.item(),
                #     "train_step": global_step,
                #     "perf_images_per_sec": imgs_per_sec,
                # }
                logger.info(
                    "epoch: %d, step: %s, loss: %.6f, imgs/sec: %.2g",
                    epoch,
                    global_step,
                    loss.item(),
                    imgs_per_sec,
                )

        # Validate
        all_y_pred, all_y_true, all_metadata = [], [], []
        for batch, (images, labels, metadata) in enumerate(
            tqdm.tqdm(test_dataloader, desc=f"Eval; epoch {epoch}")
        ):
            images = images.to(args.device)
            with ctx:
                encoded = backbone.img_encode(images)
                outputs = linear_model(encoded.img_features)

            y_pred = outputs.argmax(axis=1)
            all_y_pred.append(y_pred.cpu())
            all_y_true.append(labels)
            all_metadata.append(metadata)

        all_y_pred = torch.cat(all_y_pred, axis=0)
        all_y_true = torch.cat(all_y_true, axis=0)
        all_metadata = torch.cat(all_metadata, axis=0)

        test_metrics, _ = dataset.eval(all_y_pred, all_y_true, all_metadata)
        msg = ", ".join(f"{name}: {value:.3f}" for name, value in test_metrics.items())
        logger.info("epoch: %d, " + msg, epoch)

    # TODO: I don't know why this is so slow. 42K examples should be faster than 40 seconds.
    logger.info("Constructing examples.")
    examples = [
        interfaces.Example(
            str(i),
            float(y_pred == y_true),
            {"y_pred": y_pred.item(), "y_true": y_true.item()},
        )
        for i, (y_pred, y_true) in enumerate(zip(tqdm.tqdm(all_y_pred), all_y_true))
    ]
    logger.info("%d examples done.", len(examples))

    @beartype.beartype
    def _calc_mean_score(examples: list[interfaces.Example]) -> float:
        # Use the dataset.eval to evaluate a particular set of examples
        all_y_pred = torch.tensor([example.info["y_pred"] for example in examples])
        all_y_true = torch.tensor([example.info["y_true"] for example in examples])
        metrics, _ = dataset.eval(all_y_pred, all_y_true, None)
        return metrics["F1-macro_all"]

    return interfaces.BenchmarkReport("iWildCam", examples, {}, _calc_mean_score)
