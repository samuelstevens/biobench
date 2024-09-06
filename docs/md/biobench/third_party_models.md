Module biobench.third_party_models
==================================

Classes
-------

`OpenClip(ckpt: str, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * biobench.interfaces.VisionBackbone
    * torch.nn.modules.module.Module

    ### Methods

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> biobench.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :

`TimmVit(ckpt: str, **kwargs)`
:   Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * biobench.interfaces.VisionBackbone
    * torch.nn.modules.module.Module

    ### Methods

    `img_encode(self, batch: jaxtyping.Float[Tensor, 'batch 3 width height']) ‑> biobench.interfaces.EncodedImgBatch`
    :

    `make_img_transform(self)`
    :