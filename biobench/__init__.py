""" """

import typing

import tyro

from . import aimv2, third_party_models
from .registry import list_vision_backbones, register_vision_backbone

register_vision_backbone("timm", third_party_models.Timm)
register_vision_backbone("open-clip", third_party_models.OpenClip)
register_vision_backbone("dinov2", third_party_models.DinoV2)
register_vision_backbone("aimv2", aimv2.AIMv2)

# Some helpful types
if typing.TYPE_CHECKING:
    # Static type seen by language servers, type checkers, etc.
    ModelOrg = str
else:
    # Runtime type used by tyro.
    ModelOrg = tyro.extras.literal_type_from_choices(list_vision_backbones())
