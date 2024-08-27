import dataclasses
import typing

import beartype

from . import third_party
from .interfaces import VisionModel

Org = typing.Literal["open_clip"]


@beartype.beartype
@dataclasses.dataclass
class Params:
    org: Org = "open_clip"
    ckpt: str = "RN50/openai"


@beartype.beartype
def load_model(params: Params) -> VisionModel:
    if params.org == "open_clip":
        arch, ckpt = third_party.OpenClip.parse_model_str(params.ckpt)
        return third_party.OpenClip(arch, ckpt)
    else:
        typing.assert_never(params.org)
