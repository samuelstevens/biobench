"""
# MammalNet

We frame behavior recognition as a classification task.
Given a short video segment, embed the video via some frame-sampling strategy and associate that embedding with a label.
We train a simple nearest-centroid classifier [which works well with few-shot tasks](https://arxiv.org/abs/1911.04623) over these representation-label pairs.

You must use torchcodec 0.2 with torch 2.6.
If you have torch 2.7, then use torchcodec 0.3.

If you use this benchmark, please cite the original work:

```
@inproceedings{chen2023mammalnet,
  title={Mammalnet: A large-scale video benchmark for mammal recognition and behavior understanding},
  author={Chen, Jun and Hu, Ming and Coker, Darren J and Berumen, Michael L and Costelloe, Blair and Beery, Sara and Rohrbach, Anna and Elhoseiny, Mohamed},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13052--13061},
  year={2023}
}
```
"""

import csv
import dataclasses
import functools
import json
import logging
import os.path

import beartype
import numpy as np
import spdl.io
import spdl.pipeline
import torch
from jaxtyping import Float16, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from .. import config, helpers, registry, reporting, simpleshot

logger = logging.getLogger(__name__)


@beartype.beartype
def benchmark(cfg: config.Experiment) -> reporting.Report:
    # 1. Load model
    backbone = registry.load_vision_backbone(cfg.model)
    backbone = backbone.to(cfg.device)

    # 2. Load data.
    test_features = get_features(cfg, backbone, is_train=False)
    train_features = get_features(cfg, backbone, is_train=True)

    # 4. Do simpleshot.
    clf = init_clf(cfg)
    clf.fit(train_features.x, train_features.y)

    true_labels = test_features.y
    pred_labels = clf.predict(test_features.x)

    # Return benchmark report.
    preds = [
        reporting.Prediction(
            str(video_id),
            float(pred == true),
            {"y_pred": pred.item(), "y_true": true.item()},
        )
        for video_id, pred, true in zip(test_features.ids, pred_labels, true_labels)
    ]
    return reporting.Report("mammalnet", preds, cfg)


@beartype.beartype
def score(preds: list[reporting.Prediction]) -> float:
    return reporting.macro_f1(preds)


@beartype.beartype
def init_clf(cfg: config.Experiment):
    return simpleshot.SimpleShotClassifier(device="cuda:0")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Features:
    x: Float16[np.ndarray, "n dim"]
    y: Int[np.ndarray, " n"]
    ids: Shaped[np.ndarray, " n"]


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_features(
    cfg: config.Experiment, backbone: registry.VisionBackbone, *, is_train: bool
) -> Features:
    """
    Gets all model features and true labels for all frames and all examples in the dataloader.

    Returns it as a pair of big tensors; other tasks use a dedicated class for this, but here it's just a tuple.

    Args:
        args: KABR task arguments.
        backbone: Vision backbone.
        is_train: Whether it's training data or not.

    Returns:
    """
    img_transform = backbone.make_img_transform()
    backbone = torch.compile(backbone)
    split = "train" if is_train else "val"

    pipeline = (
        spdl.pipeline.PipelineBuilder()
        .add_source(list_files(cfg.data.mammalnet, split=split))
        .pipe(
            functools.partial(load_clip, img_transform=img_transform),
            concurrency=cfg.n_workers,
        )
        .aggregate(cfg.batch_size)
        .add_sink(4)
        .build(num_threads=cfg.n_workers, report_stats_interval=60)
    )

    all_feats, all_labels, all_ids = [], [], []

    with pipeline.auto_stop():
        for batch in helpers.progress(pipeline, desc=f"mammalnet/{split}"):
            with torch.amp.autocast(cfg.device):
                breakpoint()
                frames, labels, ids = batch
                frames = frames.to(cfg.device, non_blocking=True)
                # conv2d doesn't support multiple batch dimensions, so we have to view() before and after the model.img_encode() call.
                bsz, n_frames, c, h, w = frames.shape
                frames = frames.view(bsz * n_frames, c, h, w)
                outputs = backbone.img_encode(frames)
                features = outputs.img_features.view(bsz, n_frames, -1)

                features = aggregate_frames(features.to(torch.float16))
                all_feats.append(features.cpu())

            all_labels.append(labels.cpu())
            all_ids.extend(ids)

    all_feats = torch.cat(all_feats, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_ids = np.array(all_ids)

    return Features(all_feats, all_labels, all_ids)


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Video:
    # Document this class with docstrings for each attribute using """..""" AI!
    path: str
    vid_id: str
    species_id: int
    behavior_id: int
    n_frames: int


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    label: str
    """The class label for this annotation segment."""
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Detection:
    id: str
    """Unique identifier for the video clip."""
    taxonomy: list[dict[str, str]]
    """Taxonomic classification information for the detected animal."""
    annotations: list[Annotation]
    """List of time segments with behavior annotations."""
    duration_s: int
    """Total duration of the video in seconds."""
    resolution: tuple[int, int]
    """Video resolution in pixels"""
    fps: int
    """Frames per second of the video."""
    subset: str
    """Dataset split this video belongs to (e.g., 'train', 'val', 'test')."""
    url: str
    """Original source URL for the video."""

    @classmethod
    def from_json(cls, id, dct):
        annotations = [
            Annotation(
                label=ann["label"],
                start_s=float(ann["segment"][0]),
                end_s=float(ann["segment"][1]),
            )
            for ann in dct.pop("annotations")
        ]
        taxonomy = dct.pop("taxnomy")
        duration_s = dct.pop("duration")
        resolution = tuple(int(x) for x in dct.pop("resolution").split("x"))
        return cls(
            id=id,
            taxonomy=taxonomy,
            annotations=annotations,
            duration_s=duration_s,
            resolution=resolution,
            **dct,
        )


@beartype.beartype
def list_files(root: str, *, split: str, composition: str = "composition"):
    if not os.path.exists(root) or not os.path.isdir(root):
        msg = f"Path '{root}' doesn't exist. Did you download the MammalNet dataset?"
        raise RuntimeError(msg)

    lengths = {}
    with open(os.path.join(root, "annotation", "detection_annotations.json")) as fd:
        for key, value in json.load(fd).items():
            det = Detection.from_json(key, value)
            if len(det.annotations) == 1:
                ann = det.annotations[0]
                lengths[det.id] = int(det.fps * ann.duration_s)
            else:
                for i, ann in enumerate(det.annotations):
                    lengths[f"{det.id}_{i + 1}"] = int(det.fps * ann.duration_s)

    with open(os.path.join(root, "annotation", composition, f"{split}.csv")) as fd:
        reader = csv.reader(fd, delimiter=" ")
        for rel_path, species_id, behavior_id in reader:
            # the CSV already prefixes "trimmed_videos/..."
            full_path = os.path.join(root, *rel_path.split("/"))
            if not os.path.isfile(full_path):
                logger.warn("Missing clip '%s'; skipping", full_path)
                continue

            vid_id, ext = os.path.splitext(os.path.basename(full_path))

            yield Video(
                full_path, vid_id, int(species_id), int(behavior_id), lengths[vid_id]
            )


@beartype.beartype
def load_clip(
    video: Video, img_transform, *, n_frames: int = 32, device: str = "cuda:0"
) -> torch.Tensor:
    """
    Returns: [n_frames, C, W, H] on the requested CUDA device.
    Caller's collate -> [B, n_frames, C, W, H]
    """

    # 1. demux once
    packets = spdl.io.demux_video(video.path)

    # 2. decide which frames we need

    idx = np.linspace(0, video.n_frames - 1, n_frames, dtype=int).tolist()

    # 3. decode only those frames (CPU, RGB24)
    frames = spdl.io.sample_decode_video(packets, idx)

    # 4. frames -> numpy uint8 [T,H,W,C]
    buf = spdl.io.convert_frames(frames)
    arr = spdl.io.to_numpy(buf)  # still CPU

    # 5. PIL -> user-supplied transform
    imgs = [img_transform(Image.fromarray(img)) for img in arr]  # each [C,W,H]

    clip = torch.stack(imgs, dim=0)  # [T,C,W,H]
    return clip.to(device, non_blocking=True)


@jaxtyped(typechecker=beartype.beartype)
def aggregate_frames(
    features: Float16[Tensor, "batch n_frames dim"],
) -> Float16[Tensor, "batch dim"]:
    return torch.max(features, dim=1).values
