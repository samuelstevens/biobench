import dataclasses

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import registry


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class AIMv2Config:
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
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
    """

    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    qkv_bias: bool = False
    use_bias: bool = False


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
    def __init__(self, config: AIMv2Config):
        super().__init__()
        d = config.hidden_size
        d_mlp = config.intermediate_size

        self.fc1 = torch.nn.Linear(d, d_mlp, bias=config.use_bias)
        self.fc2 = torch.nn.Linear(d_mlp, d, bias=config.use_bias)
        self.fc3 = torch.nn.Linear(d, d_mlp, bias=config.use_bias)

    def forward(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        x = torch.nn.functional.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class PatchEmbed(torch.nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.proj = torch.nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ViTPreprocessor(torch.nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = PatchEmbed(config)
        self.pos_embed = torch.nn.Parameter(
            torch.zeros((1, num_patches, config.hidden_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


class Attention(torch.nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        dim = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.attn_drop = torch.nn.Dropout(config.attention_dropout)
        self.proj = torch.nn.Linear(dim, dim, bias=config.use_bias)
        self.proj_drop = torch.nn.Dropout(config.projection_dropout)

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


class Block(torch.nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.attn = Attention(config)
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config)
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            Block(config) for _ in range(config.num_hidden_layers)
        ])
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, object]:
        hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            tokens = block(tokens, mask)
            if output_hidden_states:
                hidden_states += (tokens,)
        tokens = self.post_trunk_norm(tokens)
        return tokens, hidden_states


@jaxtyped(typechecker=beartype.beartype)
class AIMv2(registry.VisionBackbone):
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        # Load the config from the HF repo
        import json
        config_path = download_hf_file(ckpt, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create the config object
        config = AIMv2Config(
            hidden_size=config_dict.get("hidden_size", 1024),
            intermediate_size=config_dict.get("intermediate_size", 2816),
            num_hidden_layers=config_dict.get("num_hidden_layers", 24),
            num_attention_heads=config_dict.get("num_attention_heads", 8),
            num_channels=config_dict.get("num_channels", 3),
            image_size=config_dict.get("image_size", 224),
            patch_size=config_dict.get("patch_size", 14),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            projection_dropout=config_dict.get("projection_dropout", 0.0),
            qkv_bias=config_dict.get("qkv_bias", False),
            use_bias=config_dict.get("use_bias", False)
        )
        
        self.preprocessor = ViTPreprocessor(config)
        self.trunk = Transformer(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        x = self.preprocessor(pixel_values)
        x, hidden_states = self.trunk(
            x, mask, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            return res

        return dict(
            last_hidden_state=x,
            hidden_states=hidden_states,
        )


@beartype.beartype
def download_hf_file(ckpt: str, filepath: str, *, force_download: bool = False) -> str:
    """
    Download a file from a Hugging Face model repository.
    
    Args:
        ckpt: The model checkpoint identifier (e.g., 'apple/aimv2-large-patch14-224')
        filepath: The path to the file within the repo (e.g., 'config.json')
        force_download: Whether to force download even if the file exists locally
        
    Returns:
        The path to the downloaded file on the local filesystem
    """
    import os
    import requests
    from pathlib import Path
    from . import helpers
    
    # Construct the URL
    url = f"https://huggingface.co/{ckpt}/resolve/main/{filepath}"
    
    # Create the local path
    cache_dir = helpers.get_cache_dir()
    local_dir = os.path.join(cache_dir, "hf", ckpt)
    local_path = os.path.join(local_dir, filepath)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Check if the file exists
    if os.path.exists(local_path) and not force_download:
        return local_path
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return local_path
