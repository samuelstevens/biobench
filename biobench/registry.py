"""
Stores all vision backbones.
Users can register new custom backbones from their code to evaluate on biobench using `register_vision_backbone`.
As long as it satisfies the `biobench.interfaces.VisionBackbone` interface, it will work will all tasks.

.. include:: ./tutorial.md
"""

import logging

import beartype

from . import config, interfaces

logger = logging.getLogger(__name__)

_global_backbone_registry: dict[str, type[interfaces.VisionBackbone]] = {}


@beartype.beartype
def load_vision_backbone(
    model_cfg: config.Model,
) -> interfaces.VisionBackbone:
    """
    Load a pretrained vision backbone.
    """
    if model_cfg.org not in _global_backbone_registry:
        raise ValueError(f"Org '{model_cfg.org}' not found.")

    cls = _global_backbone_registry[model_cfg.org]
    return cls(model_cfg.ckpt)


@beartype.beartype
def register_vision_backbone(model_org: str, cls: type[interfaces.VisionBackbone]):
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


_global_mllm_registry: dict[tuple[str, str], interfaces.Mllm] = {}


@beartype.beartype
def load_mllm(cfg: config.Model) -> interfaces.Mllm:
    """
    Load a multimodal LLM configuration.
    """
    key = (cfg.org, cfg.ckpt)
    if key not in _global_mllm_registry:
        raise ValueError(f"Model '{key}' not found.")

    return _global_mllm_registry[key]


@beartype.beartype
def register_mllm(model_org: str, mllm: interfaces.Mllm):
    """
    Register a new multimodal LLM configuration.
    """
    key = (model_org, mllm.name)
    if key in _global_mllm_registry:
        logger.warning("Overwriting key '%s' in registry.", key)
    _global_mllm_registry[key] = mllm


@beartype.beartype
def list_mllms() -> list[tuple[str, str]]:
    """
    List all registered multimodal LLM models.
    """
    return list(_global_mllm_registry.keys())
