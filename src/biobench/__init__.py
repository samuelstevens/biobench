""" """

import typing

import tyro

from . import aimv2, dinov3, third_party_models, vjepa
from .registry import list_vision_backbones, register_vision_backbone

register_vision_backbone("timm", third_party_models.Timm)
register_vision_backbone("open-clip", third_party_models.OpenClip)
register_vision_backbone("dinov2", third_party_models.DinoV2)
register_vision_backbone("sam2", third_party_models.SAM2)
register_vision_backbone("aimv2", aimv2.AIMv2)
register_vision_backbone("vjepa", vjepa.Vjepa)
register_vision_backbone("dinov3", dinov3.DinoV3)

# Some helpful types
if typing.TYPE_CHECKING:
    # Static type seen by language servers, type checkers, etc.
    ModelOrg = str
else:
    # Runtime type used by tyro.
    ModelOrg = tyro.extras.literal_type_from_choices(list_vision_backbones())
