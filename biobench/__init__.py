"""
Package for organizing all the code to run benchmarks.

Submodules are either tasks or helpers for organizing code.

The most important modules to understand are:

* `benchmark` because it is the launch script that runs all the tasks.
* `interfaces` because it defines how everything hooks together.
* Any of the task modules--`newt` is well documented.
* Any of the vision modules--`third_party_models.OpenClip` is highly relevant to anyone using the [open_clip](https://github.com/mlfoundations/open_clip) codebase to train models.

# Task Modules

* `ages`: Classify images of juvenile birds using only adult birds for training.
* `beluga`: Re-identification of Beluga whales using nearest neighbors.
* `birds525`: 1-shot classification with nearest-neighbor of the Kaggle Birds dataset.
* `fishnet`: Predict presense/absence of 9 different traits in images of fish.
* `iwildcam`: Species classification from camera trap imagery using multiclass ridge regression.
* `kabr`: Behavior classification of single-subject animals using simpleshot of mean frame representations.
* `newt`: 164 binary classification tasks using an binary SVM trained on image features.
* `plantnet`: long-tail classification of plant species.
* `plankton`: classification of microscopic images of phytoplankton.
# Helper Modules

* `interfaces`: All the interfaces that describe how the different modules communicate.
* `registry`: How to add/load vision models to BioBench.
* `simpleshot`: An implementation of nearest-centroid classification from [Simpleshot](https://arxiv.org/abs/1911.04623).
* `third_party_models`: Some pre-written vision backbones that load checkpoints from [open_clip](https://github.com/mlfoundations/open_clip) and [timm](https://github.com/huggingface/pytorch-image-models). You can use these as inspiration for writing your own vision backbones or check out `biobench.registry` for a tutorial.

# Future Tasks

These are tasks that I plan on adding but are not yet done.

* `rarespecies`: Waiting on a [bug](https://huggingface.co/datasets/imageomics/rare-species/discussions/8) in Huggingface's datasets library.
* [FishVista](https://github.com/Imageomics/Fish-Vista): I want to add trait classification: given patch-level features, can we train a linear probe to predict the presence/absense of a feature?

# Discussion of Task Importance

(TODO)

Main points:

* Some tasks are real tasks (KABR, Beluga).
* Others are made up (Birds525, RareSpecies).
* Some are in-between (NeWT, FishNet).
* Some are made up and we hypothesize that they predict performance on real tasks (ImageNet-1K, iNat2021).
* Likely, the real tasks are most important, but they are also the most specialized (and thus least likely to predict performance on other tasks).
* We don't know if ImageNet-1K or iNat2021 still predict performance on real, specialized tasks. They definitely did 2-5 years ago---is that still true with new models?

We can also compare tasks by the fundamental task and the types of images used.

| Name | Fundamental Task | Image Distribution |
|---|---|---|
| `ages` | Classification | object-centric in-situ RGB photos |
| `beluga` | Re-identification (via KNN) | top-down cropped RGB photos |
| `birds525` | Classification | object-centric in-situ RGB photos |
| `fishnet` | Functional trait prediction | lab specimen RGB photos |
| `imagenet` | Classification | object-centric in-situ RGB photos |
| `inat21` | Classification | object-centric in-situ RGB photos |
| `iwildcam` | Classification | camera-trap images, black-and-white |
| `kabr` | Classification (Behavior) | |
| `leopard` | Re-identification (via SVM) |
| `newt` | | object-centric in-situ RGB photos |
| `plantnet` | Classification | object-centric in-situ RGB photos |
| `plankton` | Classification | microscopic black-and-white slides |
| `rarespecies` | Classification | object-centric in-situ RGB photos |

Most of our tasks are species classification.
The only non-species classification tasks are Belgua whale and leopard re-identification, FishNet's functional trait prediction, and KABR's *behavior* classification.
The age task includes a train/test mismatch and iWildCam and the plankton task are different image distributions.

But Birds525, ImageNet, iNat21, Pl@ntNet, and RareSpecies all fill the same role.
I would expect performance on these tasks to be extremely correlated.

This raises the question: why do we have multiple tasks that fill the same role?

.. include:: ./confidence-intervals.md
"""

import typing

import tyro
from beartype.claw import beartype_this_package

from . import interfaces, third_party_models
from .registry import (
    list_vision_backbones,
    load_vision_backbone,
    register_mllm,
    register_vision_backbone,
)

beartype_this_package()

register_vision_backbone("timm-vit", third_party_models.TimmVit)
register_vision_backbone("open-clip", third_party_models.OpenClip)
register_mllm(
    "openrouter",
    interfaces.Mllm(
        "meta-llama/llama-3.2-3b-instruct", 131_000, 0.015, 0.025, ["fp32", "bf16"]
    ),
)

# Some helpful types
if typing.TYPE_CHECKING:
    # Static type seen by language servers, type checkers, etc.
    ModelOrg = str
else:
    # Runtime type used by tyro.
    ModelOrg = tyro.extras.literal_type_from_choices(list_vision_backbones())


__all__ = [
    "interfaces",
    "load_vision_backbone",
    "register_vision_backbone",
    "list_vision_backbones",
    "ModelOrg",
]
