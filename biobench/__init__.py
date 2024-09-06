"""
Package for organizing all the code to run benchmarks.

.. include:: ./confidence-intervals.md
"""

from . import interfaces, kabr, newt, third_party_models
from .registry import (
    list_vision_backbones,
    load_vision_backbone,
    register_vision_backbone,
)

register_vision_backbone("timm-vit", third_party_models.TimmVit)
register_vision_backbone("open_clip", third_party_models.OpenClip)


__all__ = [
    "interfaces",
    "load_vision_backbone",
    "register_vision_backbone",
    "list_vision_backbones",
    "newt",
    "kabr",
]
