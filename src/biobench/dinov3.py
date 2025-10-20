import dataclasses
import math
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import einops
import torch
import torch.nn.functional as F
import torch.nn.init
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from . import registry


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    img_size: int = 224
    """Image width and height in pixels."""
    patch_size: int = 16
    """Size of each patch in pixels."""
    in_chans: int = 3
    """Number of input image channels."""
    pos_embed_rope_base: float = 100.0
    """Base frequency for RoPE positional encoding."""
    pos_embed_rope_min_period: float | None = None
    """Minimum period for RoPE positional encoding."""
    pos_embed_rope_max_period: float | None = None
    """Maximum period for RoPE positional encoding."""
    pos_embed_rope_normalize_coords: tp.Literal["min", "max", "separate"] = "separate"
    """Coordinate normalization method for RoPE encoding."""
    pos_embed_rope_dtype: str = "bf16"
    """Data type for RoPE positional encoding."""
    embed_dim: int = 768
    """Embedding dimension for transformer."""
    depth: int = 12
    """Number of transformer blocks."""
    num_heads: int = 12
    """Number of attention heads."""
    ffn_ratio: float = 4.0
    """Feed-forward network expansion ratio."""
    qkv_bias: bool = True
    """Whether to use bias in QKV projection."""
    ffn_layer: str = "mlp"
    """Type of feed-forward network layer."""
    ffn_bias: bool = True
    """Whether to use bias in feed-forward network."""
    proj_bias: bool = True
    """Whether to use bias in output projection."""
    n_storage_tokens: int = 0
    """Number of storage/register tokens."""
    mask_k_bias: bool = False
    """Whether to mask K bias in attention."""
    untie_cls_and_patch_norms: bool = False
    """Whether to use separate norms for CLS and patch tokens."""
    untie_global_and_local_cls_norm: bool = False
    """Whether to use separate norms for global and local CLS tokens."""
    device: tp.Any | None = None
    """Device for tensor operations."""


@beartype.beartype
def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)

        self.img_size = image_hw
        self.patch_size = patch_hw

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw
        )

    def forward(
        self, x_bchw: Float[Tensor, "batch channnels height width"]
    ) -> Float[Tensor, "batch height_p width_p dim"]:
        x_bdhw = self.proj(x_bchw)  # B C H W
        _, _, h, w = x_bdhw.shape
        x_bhwd = einops.rearrange(x_bdhw, "b d h w -> b h w d")
        return x_bhwd


@jaxtyped(typechecker=beartype.beartype)
class RopePositionEmbedding(nn.Module):
    # RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
    # Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None,
        min_period: float | None,
        max_period: float | None,
        normalize_coords: tp.Literal["min", "max", "separate"],
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.d_head = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.dtype = dtype

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.register_buffer(
            "periods", torch.empty(self.d_head // 4, dtype=dtype), persistent=True
        )
        dd = {"device": self.periods.device, "dtype": self.dtype}
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.d_head // 4, **dd) / (self.d_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.d_head // 4, **dd)
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]

        self.periods.data = periods

    def forward(self, *, h: int, w: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_hw = max(h, w)
            coords_h = torch.arange(0.5, h, **dd) / max_hw  # [h]
            coords_w = torch.arange(0.5, w, **dd) / max_hw  # [w]
        elif self.normalize_coords == "min":
            min_hw = min(h, w)
            coords_h = torch.arange(0.5, h, **dd) / min_hw  # [h]
            coords_w = torch.arange(0.5, w, **dd) / min_hw  # [w]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, h, **dd) / h  # [h]
            coords_w = torch.arange(0.5, w, **dd) / w  # [w]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]


@jaxtyped(typechecker=beartype.beartype)
class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))

    def __call__(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * self.gamma


@jaxtyped(typechecker=beartype.beartype)
class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=0))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return F.linear(input, self.weight, masked_bias)


@jaxtyped(typechecker=beartype.beartype)
def _rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


@jaxtyped(typechecker=beartype.beartype)
def _rope(
    x: Float[Tensor, "..."], sin: Float[Tensor, "..."], cos: Float[Tensor, "..."]
) -> Float[Tensor, "..."]:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (_rotate_half(x) * sin)


@jaxtyped(typechecker=beartype.beartype)
def rope_fn(
    q_bhnd: Float[Tensor, "batch head n d_head"],
    k_bhnd: Float[Tensor, "batch head n d_head"],
    rope: Tensor | tuple[Tensor, Tensor],
) -> tuple[Float[Tensor, "batch head n d_head"], Float[Tensor, "batch head n d_head"]]:
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q_bhnd.dtype
    k_dtype = k_bhnd.dtype

    sin_pd, cos_pd = rope
    rope_dtype = sin_pd.dtype

    q_bhnd = q_bhnd.to(dtype=rope_dtype)
    k_bhnd = k_bhnd.to(dtype=rope_dtype)
    _, n_heads, n, d_head = q_bhnd.shape
    n_pos, d_head = sin_pd.shape
    prefix = n - n_pos
    assert prefix >= 0, f"Got {n} residual streams but only {n_pos} patches."

    q_prefix_bhd = q_bhnd[:, :, :prefix, :]
    q_bhpd = _rope(q_bhnd[:, :, prefix:, :], sin_pd, cos_pd)
    q_bhnd = torch.cat((q_prefix_bhd, q_bhpd), dim=2)
    k_prefix_bhd = k_bhnd[:, :, :prefix, :]
    k_bhpd = _rope(k_bhnd[:, :, prefix:, :], sin_pd, cos_pd)
    k_bhnd = torch.cat((k_prefix_bhd, k_bhpd), dim=2)

    q_bhnd = q_bhnd.to(dtype=q_dtype)
    k_bhnd = k_bhnd.to(dtype=k_dtype)

    return q_bhnd, k_bhnd


@jaxtyped(typechecker=beartype.beartype)
class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.embed_dim // cfg.num_heads
        self.scale = head_dim**-0.5

        linear_cls = LinearKMaskedBias if cfg.mask_k_bias else nn.Linear
        self.qkv = linear_cls(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.qkv_bias)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.proj_bias)

    def forward(
        self,
        x_bnd: Float[Tensor, "batch n d"],
        rope: Float[Tensor, "2 n_pos d_head"] | None = None,
    ) -> Float[Tensor, "batch n d"]:
        b, n_tok, d = x_bnd.shape
        qkv_b3nd = einops.rearrange(
            self.qkv(x_bnd),
            "batch n_tok (parts d) -> batch parts n_tok d",
            parts=3,  # [q, k, v] = 3 parts
            d=d,
        )
        qkv_b3hnd = einops.rearrange(
            qkv_b3nd,
            "batch parts n_tok (n_heads d_head) -> batch parts n_heads n_tok d_head",
            parts=3,
            n_heads=self.cfg.num_heads,
            d_head=d // self.cfg.num_heads,
        )
        q_bhnd, k_bhnd, v_bhnd = torch.unbind(qkv_b3hnd, dim=1)

        if rope is not None:
            q_bhnd, k_bhnd = rope_fn(q_bhnd, k_bhnd, rope)
        x_bhnd = torch.nn.functional.scaled_dot_product_attention(
            q_bhnd, k_bhnd, v_bhnd
        )
        x_bnd = einops.rearrange(
            x_bhnd, "batch n_heads n_tok d_head -> batch n_tok (n_heads d_head)"
        )
        x_bnd = self.proj(x_bnd)
        return x_bnd


@jaxtyped(typechecker=beartype.beartype)
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def __call__(self, x: Float[Tensor, "*batch d"]) -> Float[Tensor, "*batch d"]:
        x = self.fc1(x)
        x = F.gelu(x, approximate="none")
        x = self.fc2(x)
        return x


@jaxtyped(typechecker=beartype.beartype)
class SelfAttentionBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.ffn_layer == "mlp"

        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=1e-5)
        self.attn = SelfAttention(cfg)
        self.ls1 = LayerScale(cfg.embed_dim)

        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=1e-5)
        self.mlp = Mlp(cfg.embed_dim, int(cfg.embed_dim * cfg.ffn_ratio), cfg.embed_dim)
        self.ls2 = LayerScale(cfg.embed_dim)

    def forward(
        self, x_bnd: Float[Tensor, "batch n dim"], rope=None
    ) -> Float[Tensor, "batch n dim"]:
        x_bnd = x_bnd + self.ls1(self.attn(self.norm1(x_bnd), rope=rope))
        x_bnd = x_bnd + self.ls2(self.mlp(self.norm2(x_bnd)))

        return x_bnd


_ffn_layer_lookup = {
    "mlp": Mlp,
}


_dtype_lookup = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@jaxtyped(typechecker=beartype.beartype)
class VisionTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        assert not self.cfg.untie_cls_and_patch_norms, "Not supported"
        assert self.cfg.n_storage_tokens > 0, "Not supported"

        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, cfg.n_storage_tokens, cfg.embed_dim)
        )
        self.mask_token = nn.Parameter(torch.empty(1, cfg.embed_dim))

        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim
        )
        self.rope_embed = RopePositionEmbedding(
            cfg.embed_dim,
            num_heads=cfg.num_heads,
            base=cfg.pos_embed_rope_base,
            min_period=cfg.pos_embed_rope_min_period,
            max_period=cfg.pos_embed_rope_max_period,
            normalize_coords=cfg.pos_embed_rope_normalize_coords,
            dtype=_dtype_lookup[cfg.pos_embed_rope_dtype],
        )

        self.blocks = nn.ModuleList([SelfAttentionBlock(cfg) for i in range(cfg.depth)])

        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-5)

    def prepare_tokens_with_masks(self, x: Tensor) -> tuple[Tensor, tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        cls_token = self.cls_token + 0 * self.mask_token

        x = torch.cat(
            [cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x],
            dim=1,
        )

        return x, (H, W)

    def forward(self, x: Float[Tensor, "b c h w"]) -> dict[str, Tensor]:
        x_bnd, (h, w) = self.prepare_tokens_with_masks(x)
        rope_sincos = self.rope_embed(h=h, w=w)
        for blk in self.blocks:
            x_bnd = blk(x_bnd, rope_sincos)

        x_norm_bnd = self.norm(x_bnd)
        x_norm_cls_bd = x_norm_bnd[:, 0]
        x_norm_patch_bnd = x_norm_bnd[:, self.cfg.n_storage_tokens + 1 :]

        output = {"cls": x_norm_cls_bd, "patches": x_norm_patch_bnd}
        return output


_PRETRAINED_CFGS = {
    "dinov3_vits16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        qkv_bias=True,
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vits16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6.0,
        qkv_bias=True,
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vitb16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vitl16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        qkv_bias=True,
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=False,
    ),
    "dinov3_vitl16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vith16plus": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
    "dinov3_vit7b16": Config(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3.0,
        qkv_bias=False,
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
    ),
}


@beartype.beartype
def load(name: str, fpath: str | pathlib.Path, device="cpu") -> VisionTransformer:
    if name not in _PRETRAINED_CFGS:
        raise ValueError(f"Name '{name}' not in {list(_PRETRAINED_CFGS)}.")
    cfg = _PRETRAINED_CFGS[name]
    state_dict = torch.load(fpath, mmap=True, weights_only=True, map_location="cpu")
    with torch.device("meta"):
        model = VisionTransformer(cfg)
    model.load_state_dict(state_dict, assign=True)

    model = model.to(device)
    return model


@jaxtyped(typechecker=beartype.beartype)
class DinoV3(registry.VisionBackbone):
    def __init__(self, ckpt: str, size_px: int = 256, **kwargs):
        super().__init__()
        self.size_px = size_px

        name = self._parse_name(ckpt)
        self.model = load(name, ckpt)

        self._ckpt = name

    @staticmethod
    def _parse_name(dinov3_ckpt: str) -> str:
        name_ds, sha = pathlib.Path(dinov3_ckpt).stem.split("-")
        *name, pretrain, ds = name_ds.split("_")
        assert pretrain == "pretrain"
        return "_".join(name)

    @classmethod
    @beartype.beartype
    def normalize_model_ckpt(cls, ckpt: str) -> str:
        """Normalize DINOv3 checkpoint path to canonical name."""
        return cls._parse_name(ckpt)

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        x = self.model(batch)
        return registry.EncodedImgBatch(x["cls"], x["patches"])

    def make_img_transform(self) -> Callable:
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=self.size_px),
            v2.CenterCrop(size=(self.size_px, self.size_px)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
