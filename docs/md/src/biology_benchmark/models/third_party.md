Module src.biology_benchmark.models.third_party
===============================================

Classes
-------

`OpenClip(arch: str, ckpt: str, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * src.biology_benchmark.models.interfaces.VisionModel
    * torch.nn.modules.module.Module

    ### Static methods

    `parse_model_str(model: str) ‑> tuple[str, str]`
    :   Parse a string like 'RN50/openai' into 'RN50', 'openai' for use with the open_clip package.

    ### Methods

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> src.biology_benchmark.models.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :