import dataclasses
import json
import os
import re
import typing as tp

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
    attention_dropout: float
    hidden_act: str
    hidden_size: int
    image_size: int
    initializer_range: float
    intermediate_size: int
    is_native: bool
    mlp_bias: bool
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    projection_dropout: float
    qkv_bias: bool
    rms_norm_eps: float
    torch_dtype: str
    use_bias: bool
    use_head: bool


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
class Mlp(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        self.gate_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=cfg.mlp_bias
        )
        self.up_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=cfg.mlp_bias
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=cfg.mlp_bias
        )
        if cfg.hidden_act == "silu":
            self.act_fn = torch.nn.SiLU()
        else:
            tp.assert_never(cfg.hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@jaxtyped(typechecker=beartype.beartype)
class Embeddings(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            stride=(cfg.patch_size, cfg.patch_size),
        )
        self.rms_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = torch.nn.Embedding(n_patches, cfg.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(n_patches).expand((1, -1)), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        breakpoint()
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        pos_embed = self.position_embedding(self.position_ids)
        x = self.rms_norm(x) + pos_embed
        return x


@jaxtyped(typechecker=beartype.beartype)
class ViTPreprocessor(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.embeddings = Embeddings(cfg)

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
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.k_proj = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=config.qkv_bias
        )
        self.v_proj = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=config.qkv_bias
        )
        self.q_proj = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=config.qkv_bias
        )
        self.out_proj = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=config.qkv_bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(queries.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, embed_dim
        ).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


@jaxtyped(typechecker=beartype.beartype)
class Layer(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attention = Attention(cfg)
        self.ffn = Mlp(cfg)
        self.rms_norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rms_norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attention(self.rms_norm1(x), mask)
        x = x + self.ffn(self.rms_norm2(x))
        return x


@jaxtyped(typechecker=beartype.beartype)
class Transformer(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Layer(cfg) for _ in range(cfg.num_hidden_layers)
        ])

    def forward(self, tokens: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        for layer in self.layers:
            tokens = layer(tokens)
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
        assert cfg_dct.pop("model_type") == "aimv2_vision_model"
        cfg = Config(**cfg_dct)

        # Model
        self.embeddings = Embeddings(cfg)
        self.encoder = Transformer(cfg)
        self.rms_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Pre-trained weights
        ckpt_fpath = download_hf_file(ckpt, "model.safetensors")
        state_dict = safetensors.torch.load_file(ckpt_fpath)
        self.load_state_dict(state_dict)

        # Extract image size from checkpoint name using regex

        match = re.search(r"patch\d+-(\d+)", ckpt)
        self.size = int(match.group(1)) if match else 224  # Default to 224 if not found

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.rms_norm(x)
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
