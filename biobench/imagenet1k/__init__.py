import dataclasses
import io
import logging
import math
import warnings

import beartype
import datasets
import datasets.formatting.torch_formatter
import numpy as np
import polars as pl
import torch
from jaxtyping import Float, Float16, Int, Shaped, jaxtyped
from PIL import Image

from .. import config, helpers, linear_probing, registry, reporting

logger = logging.getLogger("imagenet1k")

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
    module=datasets.formatting.torch_formatter.__name__,
)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float16[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    backbone = registry.load_vision_backbone(cfg.model)
    test_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

    torch.cuda.empty_cache()  # Be nice to others on the machine.

    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)
    logger.info("Trained a classifier on %d examples.", len(train_features.y))

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    preds = [
        reporting.Prediction(
            str(img_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for img_id, pred, true in zip(test_features.ids, pred_labels, true_labels)
    ]

    return reporting.Report("imagenet1k", preds, cfg)


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    return reporting.micro_acc(preds)


@jaxtyped(typechecker=beartype.beartype)
def bootstrap_scores(
    df: pl.DataFrame, *, b: int = 0, rng: np.random.Generator | None = None
) -> dict[str, Float[np.ndarray, " b"]]:
    assert df.get_column("task_name").unique().to_list() == ["imagenet1k"]

    # For some reason, one of my models only has 49.3K predictions, so I only use that many. Compared to 50K it's probably fine.
    n = 49_362

    if b > 0:
        assert rng is not None, "must provide rng argument"
        i_bs = rng.integers(0, n, size=(b, n), dtype=np.int32)

    scores = {}

    correct_buf = np.zeros((b, n), dtype=bool)

    for model_ckpt in df.get_column("model_ckpt").unique().sort().to_list():
        # pull y_true and y_pred for *one* model
        y_pred = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_pred")
            .unique()
            .sort("img_id")
            .get_column("y_pred")
            .cast(pl.Int32)
            .to_numpy()
        )

        if len(y_pred) == 0:
            continue

        y_true = (
            df.filter(pl.col("model_ckpt") == model_ckpt)
            .select("img_id", "y_true")
            .unique()
            .sort("img_id")
            .get_column("y_true")
            .cast(pl.Int32)
            .to_numpy()
        )
        assert y_true.size == y_pred.size

        if b > 0:
            # bootstrap resample into pre-allocated buffers
            np.take(y_pred == y_true, i_bs, axis=0, out=correct_buf)
            scores[model_ckpt] = correct_buf.mean(axis=1)
        else:
            scores[model_ckpt] = np.array([(y_pred == y_true).mean()])

    return scores


class Transform:
    def __init__(self, img_transform):
        self._img_transform = img_transform

    def __call__(self, example):
        img_bytes = example["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        example["image"] = self._img_transform(img)
        return example


@beartype.beartype
@torch.no_grad
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    img_transform = backbone.make_img_transform()
    backbone = backbone.to(cfg.device)
    split = "train" if is_train else "validation"

    dataset = datasets.load_dataset(
        "ILSVRC/imagenet-1k",
        split=split,
        cache_dir=helpers.get_cache_dir(),
        trust_remote_code=True,
    )

    if is_train and cfg.n_train > 0:
        i = helpers.balanced_random_sample(np.array(dataset["label"]), cfg.n_train)
        assert len(i) == cfg.n_train
    else:
        i = np.arange(dataset.num_rows)

    n_workers = min(len(i), cfg.n_workers)

    # Map
    dataset = (
        dataset.cast_column("image", datasets.Image(decode=False))
        .map(lambda ex, idx: {"id": str(idx)}, with_indices=True)
        .select(i)
        .to_iterable_dataset(num_shards=n_workers)
        .map(Transform(img_transform))
        .with_format("torch")
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=n_workers,
        drop_last=False,
        shuffle=False,
    )

    all_features, all_labels, all_ids = [], [], []

    def probe(batch):
        imgs = batch["image"].to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device):
            backbone.img_encode(imgs).img_features

    with helpers.auto_batch_size(dataloader, probe=probe) as batch_size:
        backbone = torch.compile(backbone)

        total = max(n_workers, math.ceil(len(i) / batch_size))
        it = iter(dataloader)

        logger.info("Need to embed %d batches of %d images.", total, batch_size)

        for b in helpers.progress(range(total), every=10, desc=f"in1k/{split}"):
            batch = next(it)

            images = batch["image"].to(cfg.device, non_blocking=True)

            with torch.amp.autocast(cfg.device):
                features = backbone.img_encode(images).img_features

            all_features.append(features.cpu())
            all_labels.extend(batch["label"])
            all_ids.extend(batch["id"])

    all_features = torch.cat(all_features, dim=0).cpu().to(torch.float16).numpy()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels).numpy()
    assert len(all_ids) == len(i) or cfg.n_train < 0
    logger.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    clf = linear_probing.LinearProbeClassifier(device=cfg.device)
    return clf
