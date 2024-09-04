import dataclasses
import typing

import beartype

from . import interfaces, third_party_models

Org = typing.Literal["open_clip"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Params:
    org: Org = "open_clip"
    """Where to load models from."""
    ckpt: str = "RN50/openai"
    """The org-specific string. Will error if you pass the wrong one."""


@beartype.beartype
def load_model(params: Params) -> interfaces.VisionBackbone:
    if params.org == "open_clip":
        arch, ckpt = third_party_models.OpenClip.parse_model_str(params.ckpt)
        return third_party_models.OpenClip(arch, ckpt)
    else:
        typing.assert_never(params.org)
