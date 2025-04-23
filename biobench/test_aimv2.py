import pytest
import torch
import transformers

from . import config, helpers, registry

CKPTS = [
    "apple/aimv2-large-patch14-224",
    "apple/aimv2-large-patch14-224-distilled",
    "apple/aimv2-1B-patch14-224",
    "apple/aimv2-large-patch14-448",
]
DTYPE = torch.float32
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(scope="session", params=CKPTS)
def models(request):
    ckpt = request.param
    hf = transformers.AutoModel.from_pretrained(
        ckpt, trust_remote_code=True, cache_dir=helpers.get_cache_dir()
    ).eval()
    bio = registry.load_vision_backbone(config.Model("aimv2", ckpt)).eval().to(DTYPE)
    return hf, bio


def _rand(batch: int = 1):
    torch.manual_seed(0)
    return torch.rand(batch, 3, 224, 224, dtype=DTYPE)


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
