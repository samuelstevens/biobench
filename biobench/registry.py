"""
Stores all vision backbones.
Users can register new custom backbones from their code to evaluate on biobench using `register_vision_backbone`.
As long as it satisfies the `biobench.cvml.VisionBackbone` interface, it will work will all tasks.

.. include:: ./tutorial.md
"""

import logging

import beartype

from . import config, cvml

logger = logging.getLogger(__name__)

_global_backbone_registry: dict[str, type[cvml.VisionBackbone]] = {}


@beartype.beartype
def load_vision_backbone(
    model_cfg: config.Model,
) -> cvml.VisionBackbone:
    """
    Load a pretrained vision backbone.
    """
    if model_cfg.org not in _global_backbone_registry:
        raise ValueError(f"Org '{model_cfg.org}' not found.")

    cls = _global_backbone_registry[model_cfg.org]
    return cls(model_cfg.ckpt)


@beartype.beartype
def register_vision_backbone(model_org: str, cls: type[cvml.VisionBackbone]):
    """
    Register a new vision backbone class.
    """
    if model_org in _global_backbone_registry:
        logger.warning("Overwriting key '%s' in registry.", model_org)
    _global_backbone_registry[model_org] = cls


@beartype.beartype
def list_vision_backbones() -> list[str]:
    """
    List all vision backbone model orgs.
    """
    return list(_global_backbone_registry.keys())
