"""
Package for organizing all the code to run benchmarks.
"""

import typing

import beartype

from . import interfaces, kabr, newt, third_party_models

__all__ = ["interfaces", "load_vision_backbone", "newt", "kabr"]


@beartype.beartype
def load_vision_backbone(
    args: interfaces.VisionBackboneArgs,
) -> interfaces.VisionBackbone:
    if args.org == "open_clip":
        return third_party_models.OpenClip(args.ckpt)
    elif args.org == "timm-vit":
        return third_party_models.TimmViT(args.ckpt)
    else:
        typing.assert_never(args.org)
