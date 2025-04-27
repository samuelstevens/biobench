"""
# FishNet: Fish Recognition, Detection, and Functional Traits Prediction

FishNet ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf), [code](https://github.com/faixan-khan/FishNet)) is a large-scale diverse dataset containing 94,532 images from 17,357 aquatic species.
It contains three benchmarks: fish classification, fish detection, and functional traits prediction.

We mainly focus on the third task.
We train an two-layer MLP on the visual features extracted by different model backbones to predict the presence or absence of 9 different traits.

If you use this evaluation, be sure to cite the original work:

```
@inproceedings{fishnet,
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

import logging
import os.path

import beartype
import numpy as np
import polars as pl
import sklearn.metrics
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting

logger = logging.getLogger("fishnet")

batch_size = 1024
learning_rate = 3e-4
n_steps = 30_000
threshold = 0.5


@jaxtyped(typechecker=beartype.beartype)
class Features(torch.utils.data.Dataset):
    """
    A dataset of learned features (dense vectors).
    """

    x: Float[Tensor, " n dim"]
    """Dense feature vectors from a vision backbone."""
    y: Int[Tensor, " n 9"]
    """0/1 labels of absence/presence of 9 different traits."""
    ids: list[str]
    """Image ids."""

    def __init__(
        self,
        x: Float[Tensor, " n dim"],
        y: Int[Tensor, " n n_classes"],
        ids: list[str],
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
def init_clf(input_dim: int) -> torch.nn.Module:
    """A simple MLP classifier consistent with the design in FishNet."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 9),
    )


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    """
    Calculate the macro-averaged F1 score across all fish trait predictions.

    For each fish image, we predict 9 binary traits:

    1. Feeding Path (benthic/pelagic)
    2. Tropical habitat (yes/no)
    3. Temperate habitat (yes/no)
    4. Subtropical habitat (yes/no)
    5. Boreal habitat (yes/no)
    6. Polar habitat (yes/no)
    7. Freshwater habitat (yes/no)
    8. Saltwater habitat (yes/no)
    9. Brackish water habitat (yes/no)

    The macro-averaging:

    1. Calculates an F1 score for each trait independently
    2. Takes the unweighted mean of these 9 F1 scores

    This ensures each trait contributes equally to the final score, regardless of class imbalance in the dataset (e.g., if there are many more tropical fish than brackish water fish).

    Args:
        preds: List of predictions, each containing:
            - info["y_pred"]: List of 9 binary predictions
            - info["y_true"]: List of 9 binary ground truth values

    Returns:
        The macro-averaged F1 score across all 9 traits
    """
    y_pred = np.array([pred.info["y_pred"] for pred in preds])
    y_true = np.array([pred.info["y_true"] for pred in preds])
    return sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )


def infinite(dataloader):
    """Creates an infinite iterator from a dataloader by creating a new iterator each time the previous one is exhausted.

    Args:
        dataloader: A PyTorch dataloader or similar iterable

    Yields:
        Batches from the dataloader, indefinitely
    """
    while True:
        # Create a fresh iterator from the dataloader
        it = iter(dataloader)
        for batch in it:
            yield batch


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    """
    The FishNet benchmark.
    """
    # 1. Load model.
    backbone = registry.load_vision_backbone(cfg.model)

    # 2. Get features.
    train_dataset = get_features(cfg, backbone, is_train=True)
    test_dataset = get_features(cfg, backbone, is_train=False)

    # 3. Set up classifier.
    classifier = init_clf(train_dataset.dim).to(cfg.device)

    # 4. Load datasets for classifier.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 5. Fit the classifier.
    it = infinite(train_loader)
    for step in range(n_steps):
        features, labels, _ = next(it)
        features = features.to(cfg.device)
        labels = labels.to(cfg.device, dtype=torch.float)
        output = classifier(features)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate the classifier.
        if (step + 1) % 1_000 == 0:
            preds = predict(cfg, classifier, test_loader)
            logger.info(
                "Step %d/%d (%.1f%%): %.3f (macro F1)",
                step + 1,
                n_steps,
                (step + 1) / n_steps * 100,
                score(preds),
            )
    preds = predict(cfg, classifier, test_loader)

    return reporting.Report("fishnet", preds, cfg)


@beartype.beartype
def predict(
    cfg: config.Experiment, classifier: torch.nn.Module, dataloader
) -> list[reporting.Prediction]:
    """
    Evaluates the trained classifier on a test split.

    Returns:
        List of `reporting.Prediction`s
    """
    total = 2 if cfg.debug else len(dataloader)
    it = iter(dataloader)
    preds = []
    for b in range(total):
        features, labels, ids = next(it)
        features = features.to(cfg.device)
        labels = labels.numpy()
        with torch.no_grad():
            pred_logits = classifier(features)
        pred_logits = (pred_logits > threshold).cpu().numpy()
        for id, pred, true in zip(ids, pred_logits, labels):
            info = {"y_pred": pred.tolist(), "y_true": true.tolist()}
            preds.append(reporting.Prediction(id, (pred == true).mean().item(), info))

    return preds


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    """Extract visual features."""
    if not os.path.isdir(cfg.data.fishnet):
        msg = f"Path '{cfg.data.fishnet}' doesn't exist. Did you download the FishNet dataset? See the docstring at the top of this file for instructions."
        raise ValueError(msg)

    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone.to(cfg.device))

    file = "train.csv" if is_train else "test.csv"
    dataset = ImageDataset(cfg.data.fishnet, file, transform=img_transform)
    if is_train and cfg.n_train > 0:
        i = np.random.default_rng(seed=cfg.seed).choice(
            len(dataset), cfg.n_train, replace=False, shuffle=False
        )
        assert len(i) == cfg.n_train
        dataset = torch.utils.data.Subset(dataset, i)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
    )

    def probe(batch):
        imgs = batch["img"].to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            _ = backbone.img_encode(imgs).img_features  # forward only

    all_features, all_labels, all_ids = [], [], []

    with helpers.auto_batch_size(cfg, dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), every=10, desc=f"fish/{file}"):
            images, labels, ids = next(it)
            images = images.to(cfg.device)

            features = backbone.img_encode(images).img_features
            all_features.append(features.cpu())
            all_labels.append(labels)

            all_ids.extend(ids)

    # Keep the Tensor data type for subsequent training
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    assert len(all_ids) == len(dataset)
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@jaxtyped(typechecker=beartype.beartype)
class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset for CV+ML that loads the required attribute labels.
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
        self, index
    ) -> tuple[Float[Tensor, "3 width height"], Int[Tensor, "9"], str]:
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
