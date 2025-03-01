""" """

from . import cvml, third_party_models

cvml.register_vision_backbone("timm-vit", third_party_models.TimmVit)
cvml.register_vision_backbone("open-clip", third_party_models.OpenClip)
