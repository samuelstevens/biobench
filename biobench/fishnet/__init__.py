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


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
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
    """
    assert df.get_column("task_name").unique().to_list() == ["fishnet"]

    n, *rest = df.group_by("model_ckpt").agg(n=pl.len()).get_column("n").to_list()
    assert all(n == i for i in rest)

    if b > 0:
        assert rng is not None, "must provide rng argument"
        i_bs = rng.integers(0, n, size=(b, 9, n), dtype=np.int32)

    scores = {}

    y_pred_buf = np.empty((b, 9, n), dtype=np.int32)
    y_true_buf = np.empty((b, 9, n), dtype=np.int32)

    for model_ckpt in helpers.progress(
        df.get_column("model_ckpt").unique().sort().to_list(),
        desc="fishnet/bootstrap",
        every=3,
    ):
        # pull y_true and y_pred for *one* model
        y_pred = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_pred")
            .unique()
            .sort("img_id")
            .get_column("y_pred")
            .str.json_decode()
            .to_numpy()
        )
        y_pred = np.stack(y_pred).astype(np.int32).T

        if len(y_pred) == 0:
            continue

        y_true = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_true")
            .unique()
            .sort("img_id")
            .get_column("y_true")
            .str.json_decode()
            .to_numpy()
        )
        y_true = np.stack(y_true).astype(np.int32).T

        assert y_true.size == y_pred.size

        if b > 0:
            # bootstrap resample into pre-allocated buffers
            np.take(y_pred, i_bs, axis=None, out=y_pred_buf)
            np.take(y_true, i_bs, axis=None, out=y_true_buf)
            score = reporting.macro_f1_batch(y_true_buf, y_pred_buf).mean(axis=-1)
            scores[model_ckpt] = score
        else:
            f1s = reporting.macro_f1_batch(y_true, y_pred)
            scores[model_ckpt] = f1s.mean(keepdims=True)

    return scores


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
    it = helpers.infinite(train_loader)
    for step in helpers.progress(range(n_steps), every=n_steps // 300, desc="sgd"):
        features, labels, _ = next(it)
        features = features.to(cfg.device)
        labels = labels.to(cfg.device, dtype=torch.float)
        output = classifier(features)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

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
    preds = []
    for batch in dataloader:
        features, labels, ids = batch
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

    @beartype.beartype
    def debug_cuda_mem(tag: str):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: %d", tag, torch.cuda.memory_allocated())

    def probe(batch):
        imgs, labels, ids = batch
        imgs = imgs.to(cfg.device, non_blocking=True)
        _ = backbone.img_encode(imgs).img_features  # forward only

    all_features, all_labels, all_ids = [], [], []

    with helpers.auto_batch_size(dataloader, probe=probe):
        total = len(dataloader) if not cfg.debug else 2
        it = iter(dataloader)
        for b in helpers.progress(range(total), every=10, desc=f"fishnet/{file}"):
            debug_cuda_mem("loop start")
            images, labels, ids = next(it)
            debug_cuda_mem("after batch")
            images = images.to(cfg.device)
            debug_cuda_mem("imgs.to(device)")

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

    @property
    def labels(self) -> Int[np.ndarray, "n 9"]:
        return (
            self.df.select(pl.nth(self.label_cols))
            .with_columns(
                FeedingPath=pl.when(pl.col("FeedingPath") == "benthic")
                .then(0)
                .otherwise(1)
            )
            .to_numpy()
        )
