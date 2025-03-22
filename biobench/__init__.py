""" """

import typing

import tyro
from beartype.claw import beartype_this_package

from . import third_party_models
from .registry import list_vision_backbones, register_vision_backbone

beartype_this_package()

register_vision_backbone("timm-vit", third_party_models.TimmVit)
register_vision_backbone("open-clip", third_party_models.OpenClip)

# Some helpful types
if typing.TYPE_CHECKING:
    # Static type seen by language servers, type checkers, etc.
    ModelOrg = str
else:
    # Runtime type used by tyro.
    ModelOrg = tyro.extras.literal_type_from_choices(list_vision_backbones())
