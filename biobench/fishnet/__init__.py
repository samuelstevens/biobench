"""
# FishNet: Fish Recognition, Detection, and Functional Traits Prediction

FishNet ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf), [code](https://github.com/faixan-khan/FishNet)) is a large-scale diverse dataset containing 94,532 images from 17,357 aquatic species.
It contains three benchmarks: fish classification, fish detection, and functional traits prediction.

We mainly focus on the third task.
We train an two-layer MLP on the visual features extracted by different model backbones to predict the presence or absence of 9 different traits.

If you use this evaluation, be sure to cite the original work:

```
@InProceedings{Khan_2023_ICCV,
    author    = {Khan, Faizan Farooq and Li, Xiang and Temple, Andrew J. and Elhoseiny, Mohamed},
    title     = {FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Trait Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20496-20506}
}
```

This task was contributed by [Jianyang Gu](https://vimar-gu.github.io/).
"""

import dataclasses
import logging
import os.path

import beartype
import numpy as np
import polars as pl
import sklearn
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from biobench import helpers, interfaces, registry

logger = logging.getLogger("fishnet")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args(interfaces.TaskArgs):
    """FishNet task arguments."""

    batch_size: int = 256
    """Batch size for deep model and MLP classifier."""
    n_workers: int = 4
    """number of dataloader worker processes."""
    log_every: int = 10
    """how often (number of epochs) to log progress."""
    n_epochs: int = 100
    """How many epochs to train the MLP classifier."""
    learning_rate: float = 5e-4
    """The learning rate for training the MLP classifier."""
    threshold: float = 0.5
    """The threshold to predicted "presence" rather than "absence"."""


@jaxtyped(typechecker=beartype.beartype)
class Features(torch.utils.data.Dataset):
    """
    A dataset of learned features (dense vectors).
    """

    x: Float[Tensor, " n dim"]
    """Dense feature vectors from a vision backbone."""
    y: Int[Tensor, " n 9"]
    """0/1 labels of absence/presence of 9 different traits."""
    ids: Shaped[np.ndarray, " n"]
    """Image ids."""

    def __init__(
        self,
        x: Float[Tensor, " n dim"],
        y: Int[Tensor, " n n_classes"],
        ids: Shaped[np.ndarray, " n"],
    ):
        self.x = x
        self.y = y
        self.ids = ids

    @property
    def dim(self) -> int:
        """Dimension of the dense feature vectors."""
        _, dim = self.x.shape
        return dim

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, index
    ) -> tuple[Float[Tensor, " dim"], Int[Tensor, " n_classes"], str]:
        return self.x[index], self.y[index], self.ids[index]


@beartype.beartype
def init_classifier(input_dim: int) -> torch.nn.Module:
    """A simple MLP classifier consistent with the design in FishNet."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 9),
    )


@beartype.beartype
def calc_macro_f1(examples: list[interfaces.Example]) -> float:
    """TODO: docs."""
    y_pred = np.array([example.info["y_pred"] for example in examples])
    y_true = np.array([example.info["y_true"] for example in examples])
    score = sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )
    return score.item()


@beartype.beartype
def benchmark(
    args: Args, model_args: interfaces.ModelArgs
) -> tuple[interfaces.ModelArgs, interfaces.TaskReport]:
    """
    The FishNet benchmark.
    """
    # 1. Load model.
    backbone = registry.load_vision_backbone(*model_args)

    # 2. Get features.
    train_dataset = get_features(args, backbone, is_train=True)
    test_dataset = get_features(args, backbone, is_train=False)

    # 3. Set up classifier.
    classifier = init_classifier(train_dataset.dim).to(args.device)

    # 4. Load datasets for classifier.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 5. Fit the classifier.
    for epoch in range(args.n_epochs):
        total = 2 if args.debug else len(train_loader)
        it = iter(train_loader)
        for b in range(total):
            features, labels, _ = next(it)
            features = features.to(args.device)
            labels = labels.to(args.device, dtype=torch.float)
            output = classifier(features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate the classifier.
        if (epoch + 1) % args.log_every == 0:
            examples = evaluate(args, classifier, test_loader)
            score = calc_macro_f1(examples)
            logger.info("Epoch %d/%d: %.3f", epoch + 1, args.n_epochs, score)

    return model_args, interfaces.TaskReport(
        "FishNet", examples, calc_mean_score=calc_macro_f1
    )


@beartype.beartype
def evaluate(
    args: Args, classifier: torch.nn.Module, dataloader
) -> list[interfaces.Example]:
    """
    Evaluates the trained classifier on a test split.

    Returns:
        a list of Examples.
    """
    total = 2 if args.debug else len(dataloader)
    it = iter(dataloader)
    examples = []
    for b in range(total):
        features, labels, ids = next(it)
        features = features.to(args.device)
        labels = labels.numpy()
        ids = ids.numpy()
        with torch.no_grad():
            pred_logits = classifier(features)
        pred_logits = (pred_logits > args.threshold).cpu().numpy()
        for id, pred, true in zip(ids, pred_logits, labels):
            example = interfaces.Example(
                str(id),
                float((pred == true).all()),
                {"y_pred": pred.tolist(), "y_true": true.tolist()},
            )
            examples.append(example)

    return examples


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    args: Args, backbone: interfaces.VisionBackbone, *, is_train: bool
) -> Features:
    """Extract visual features."""
    if not os.path.isdir(args.datadir):
        msg = f"Path '{args.datadir}' doesn't exist. Did you download the FishNet dataset? See the docstring at the top of this file for instructions. If you did download it, pass the path with '--fishnet-args.datadir'; see --help for more."
        raise ValueError(msg)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(args.device))

    file = "train.csv" if is_train else "test.csv"
    dataset = ImageDataset(args.datadir, file, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.n_workers
    )

    all_features, all_labels, all_ids = [], [], []

    total = 2 if args.debug else len(dataloader)
    it = iter(dataloader)
    for b in helpers.progress(range(total), every=args.log_every, desc=file):
        images, labels, _ = next(it)
        images = images.to(args.device)

        features = backbone.img_encode(images).img_features
        all_features.append(features.cpu())
        all_labels.append(labels)

        ids = np.arange(len(labels)) + b * args.batch_size
        all_ids.append(ids)

    # Keep the Tensor data type for subsequent training
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_ids = np.concatenate(all_ids, axis=0)

    return Features(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset that loads the required attribute labels.
    """

    def __init__(self, root_dir: str, csv_file: str, transform):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath",
            "Tropical",
            "Temperate",
            "Subtropical",
            "Boreal",
            "Polar",
            "freshwater",
            "saltwater",
            "brackish",
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())
        self.transform = transform

        # Corresponding column indices
        self.image_col = 4
        self.folder_col = 13
        self.label_cols = [15, 16, 17, 18, 19, 20, 21, 22, 23]
        logger.info("csv file: %s has %d item.", csv_file, len(self.df))

    def __getitem__(
        self, index: int
    ) -> tuple[Float[Tensor, "3 width height"], Float[Tensor, "9"], str]:
        row_data = self.df.row(index)
        image_name = row_data[self.image_col]
        image_name = image_name.split("/")[-1]
        folder = row_data[self.folder_col]
        image_path = os.path.join(self.root_dir, "Image_Library", folder, image_name)
        image = Image.open(image_path)

        # Extract the required attribute labels.
        label = []
        for col in self.label_cols:
            value = row_data[col]
            if col == 15:
                if value == "pelagic":
                    value = 1
                elif value == "benthic":
                    value = 0
                else:
                    raise ValueError("FeedingPath can only be pelagic or benthic.")
            label.append(value)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

    def __len__(self) -> int:
        return len(self.df)
