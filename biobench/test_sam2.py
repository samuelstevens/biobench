import pytest
import torch

from . import config, registry

DTYPE = torch.float32
ATOL, RTOL = 1e-5, 1e-4

CKPTS = ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus"]


@pytest.fixture(scope="session", params=CKPTS, ids=lambda x: x[0])
def models(request):
    model_ckpt = request.param
    ref = None
    ours = registry.load_vision_backbone(config.Model("sam2", model_ckpt))
    return ref, ours


def test_is_nn(models):
    ref, ours = models
    assert isinstance(ours, torch.nn.Module)
