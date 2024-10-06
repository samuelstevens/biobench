"""
Package for organizing all the code to run benchmarks.

Submodules are either tasks (like `biobench.kabr`, `biobench.birds525`) or helpers for organizing code (like `biobench.registry`, `biobench.interfaces`).

The most important modules to understand are:

* `benchmark` because it is the launch script that runs all the tasks.
* `biobench.interfaces` because it defines how everything hooks together.
* Any of the task modules--`biobench.newt` is well documented.
* Any of the vision modules--`biobench.third_party_models.OpenClip` is highly relevant to anyone using the [open_clip](https://github.com/mlfoundations/open_clip) codebase to train models.

## Task Modules

* `biobench.birds525`: 1-shot classification with nearest-neighbor of the Kaggle Birds dataset.
* `biobench.iwildcam`: Species classification using multiclass linear regression.
* `biobench.kabr`: Behavior classification of single-subject animals using simpleshot of mean frame representations.
* `biobench.newt`: 164 binary classification tasks using an binary SVM trained on image features.

## Helper Modules

* `biobench.interfaces`:
* `biobench.registry`:
* `biobench.simpleshot`: An implementation of nearest-centroid classification from [Simpleshot](https://arxiv.org/abs/1911.04623).
* `biobench.third_party_models`:

## Future Tasks

These are tasks that I plan on adding but are not yet done.

* `biobench.beluga`:
* `biobench.iwildcam`:
* `biobench.rarespecies`:
* [FishVista](https://github.com/Imageomics/Fish-Vista): I want to add trait classification: given patch-level features, can we train a linear probe to predict the presence/absense of a feature?

.. include:: ./confidence-intervals.md
"""

import typing

import tyro

from . import interfaces, kabr, newt, third_party_models
from .registry import (
    list_vision_backbones,
    load_vision_backbone,
    register_vision_backbone,
)

register_vision_backbone("timm-vit", third_party_models.TimmVit)
register_vision_backbone("open-clip", third_party_models.OpenClip)

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
    "newt",
    "kabr",
    "ModelOrg",
]
