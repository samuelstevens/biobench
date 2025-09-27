# biobench/test_dinov3.py
"""
Compares the inference results from:

* Official DINOv3 repo
* This repo's implementation

The goal is to make sure they are identical on your particular hardware.
"""

import pytest
import torch
import transformers

from biobench import config, helpers, registry

CKPTS = [
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
]
DTYPE = torch.float32
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(scope="session", params=CKPTS)
def models(request):
    ckpt = request.param
    hf = transformers.AutoModel.from_pretrained(
        ckpt, trust_remote_code=True, cache_dir=helpers.get_cache_dir()
    ).eval()
    bio = registry.load_vision_backbone(config.Model("dinov3", ckpt)).eval().to(DTYPE)
    return hf, bio


def _rand(batch: int = 1, shape: tuple[int, int] = (224, 224)):
    torch.manual_seed(0)
    return torch.rand(batch, 3, *shape, dtype=DTYPE)


def test_same_shape_single(models):
    hf, bio = models
    batch = _rand()
    h = hf(batch).last_hidden_state
    b = bio.img_encode(batch).patch_features
    assert h.shape == b.shape


def test_values_close_single(models):
    hf, bio = models
    batch = _rand()
    h = hf(batch).last_hidden_state
    b = bio.img_encode(batch).patch_features
    assert torch.allclose(h, b, atol=ATOL, rtol=RTOL)


def test_values_close_batch(models):
    hf, bio = models
    batch = _rand(batch=4)
    h = hf(batch).last_hidden_state
    b = bio.img_encode(batch).patch_features
    assert torch.allclose(h, b, atol=ATOL, rtol=RTOL)
