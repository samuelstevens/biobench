Module biobench.registry
========================
Stores all vision backbones.
Users can register new custom backbones from their code to evaluate on biobench using `register_vision_backbone`.
As long as it satisfies the `biobench.interfaces.VisionBackbone` interface, it will work will all tasks.

Functions
---------

`list_vision_backbones() ‑> list[str]`
:   List all vision backbone model orgs.

`load_vision_backbone(model_org: str, ckpt: str) ‑> biobench.interfaces.VisionBackbone`
:   Load a pretrained vision backbone.

`register_vision_backbone(model_org: str, cls: type[biobench.interfaces.VisionBackbone])`
:   Register a new vision backbone class.