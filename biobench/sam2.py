from . import registry


class SAM2(registry.VisionBackbone):
    def __init__(self, ckpt: str, **kwargs):
        self.ckpt = ckpt
