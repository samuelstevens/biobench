import dataclasses
import json
import os

import beartype
import requests
import safetensors.torch
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import helpers, registry


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """This is the configuration class to store the configuration of an `AIMv2Model`.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the [apple/aimv2-large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224).

    Args:
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the SwiGLU representations.
        num_hidden_layers: Number of hidden layers in the Transformer.
        num_attention_heads: Number of attention heads for each attention layer in the Transformer.
        num_channels: Number of input channels.
        image_size: Image size.
        patch_size: Patch size.
        rms_norm_eps: Epsilon value used for the RMS normalization layer.
        attention_dropout: Dropout ratio for attention probabilities.
        projection_dropout: Dropout ratio for the projection layer after the attention.
        torch_dtype: Data type.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    rms_norm_eps: float
    attention_dropout: float
    projection_dropout: float
    torch_dtype: str
    qkv_bias: bool
    use_bias: bool


@beartype.beartype
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


@jaxtyped(typechecker=beartype.beartype)
class SwiGLUFFN(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.fc1 = torch.nn.Linear(
            cfg.hidden_size, cfg.intermediate_size, bias=cfg.use_bias
        )
        self.fc2 = torch.nn.Linear(
            cfg.intermediate_size, cfg.hidden_size, bias=cfg.use_bias
        )
        self.fc3 = torch.nn.Linear(
            cfg.hidden_size, cfg.intermediate_size, bias=cfg.use_bias
        )

    def forward(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        x = torch.nn.functional.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbed(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = torch.nn.Conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            stride=(cfg.patch_size, cfg.patch_size),
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


@jaxtyped(typechecker=beartype.beartype)
class ViTPreprocessor(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        num_patches = (cfg.image_size // cfg.patch_size) ** 2

        self.patchifier = PatchEmbed(cfg)
        self.pos_embed = torch.nn.Parameter(
            torch.zeros((1, num_patches, cfg.hidden_size))
        )

    def forward(
        self, x: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches d"]:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


@jaxtyped(typechecker=beartype.beartype)
class Attention(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dim = cfg.hidden_size

        self.num_heads = cfg.num_attention_heads
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=cfg.qkv_bias)
        self.attn_drop = torch.nn.Dropout(cfg.attention_dropout)
        self.proj = torch.nn.Linear(dim, dim, bias=cfg.use_bias)
        self.proj_drop = torch.nn.Dropout(cfg.projection_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@jaxtyped(typechecker=beartype.beartype)
class Block(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm_1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUFFN(cfg)
        self.norm_2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


@jaxtyped(typechecker=beartype.beartype)
class Transformer(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            Block(cfg) for _ in range(cfg.num_hidden_layers)
        ])
        self.post_trunk_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, tokens: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.post_trunk_norm(tokens)
        return tokens


@jaxtyped(typechecker=beartype.beartype)
class AIMv2(registry.VisionBackbone):
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()

        # Config
        with open(download_hf_file(ckpt, "config.json"), "r") as fd:
            cfg_dct = json.load(fd)
        for key in ("architectures", "auto_map", "transformers_version"):
            cfg_dct.pop(key)
        assert cfg_dct.pop("model_type") == "aimv2"
        cfg = Config(**cfg_dct)

        # Model
        self.preprocessor = ViTPreprocessor(cfg)
        self.trunk = Transformer(cfg)

        # Pre-trained weights
        ckpt_fpath = download_hf_file(ckpt, "model.safetensors")
        state_dict = safetensors.torch.load_file(ckpt_fpath)
        self.load_state_dict(state_dict)

        # Extract image size from checkpoint name using regex
        import re

        match = re.search(r"patch\d+-(\d+)", ckpt)
        self.size = int(match.group(1)) if match else 224  # Default to 224 if not found

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x = self.preprocessor(x)
        x = self.trunk(x)
        return x

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        x = self.forward(batch)
        return registry.EncodedImgBatch(x.max(dim=1).values, x)

    def make_img_transform(self):
        import torch
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=self.size),
            v2.CenterCrop(size=(self.size, self.size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])


@beartype.beartype
def download_hf_file(ckpt: str, filepath: str, *, force: bool = False) -> str:
    """
    Download a file from a Hugging Face model repository.

    Args:
        ckpt: The model checkpoint identifier (e.g., 'apple/aimv2-large-patch14-224')
        filepath: The path to the file within the repo (e.g., 'config.json')
        force: Whether to force download even if the file exists locally

    Returns:
        The path to the downloaded file on the local filesystem
    """

    # Construct the URL
    url = f"https://huggingface.co/{ckpt}/resolve/main/{filepath}"

    # Create the local path
    cache_dir = helpers.get_cache_dir()
    local_dir = os.path.join(cache_dir, "hf", ckpt)
    local_path = os.path.join(local_dir, filepath)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Check if the file exists
    if os.path.exists(local_path) and not force:
        return local_path

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path
