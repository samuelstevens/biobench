Module biobench.models
======================

Functions
---------

`load_model(params: biobench.models.Params) ‑> biobench.interfaces.VisionBackbone`
:   

Classes
-------

`Params(org: Literal['open_clip'] = 'open_clip', ckpt: str = 'RN50/openai')`
:   Params(org: Literal['open_clip'] = 'open_clip', ckpt: str = 'RN50/openai')

    ### Class variables

    `ckpt: str`
    :   The org-specific string. Will error if you pass the wrong one.

    `org: Literal['open_clip']`
    :   Where to load models from.