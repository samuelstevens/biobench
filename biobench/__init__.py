"""
Package for organizing all the code to run benchmarks.

Submodules are either tasks (like `biobench.kabr`, `biobench.birds525`) or helpers for organizing code (like `biobench.registry`, `biobench.interfaces`).

The most important modules to understand are:

* `benchmark` because it is the launch script that runs all the tasks.
* `biobench.interfaces` because it defines how everything hooks together.
* Any of the task modules--`biobench.kabr` is well documented.
* Any of the vision modules--`biobench.third_party_models.OpenClip` is highly relevant to anyone using the [open_clip](https://github.com/mlfoundations/open_clip) codebase to train models.

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
