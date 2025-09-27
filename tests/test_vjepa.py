import pytest
import torch

from biobench import config, registry

DTYPE = torch.float32
ATOL, RTOL = 1e-5, 1e-4

CKPTS = ["vitl16", "vith16", "vith16-384"]


@pytest.fixture(scope="session", params=CKPTS, ids=lambda x: x[0])
def model(request):
    model_ckpt = request.param
    return registry.load_vision_backbone(config.Model("vjepa", model_ckpt))


def rand(b: int = 2):
    torch.manual_seed(0)
    return torch.rand(b, 3, 224, 224, dtype=DTYPE)


def test_is_nn(model):
    assert isinstance(model, torch.nn.Module)


def test_smoke(model):
    x = rand()
    model.img_encode(x.squeeze(1))
