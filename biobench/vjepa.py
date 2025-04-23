"""
# V‑JEPA *frozen‑feature* backbone port

This file is a self‑contained re‑implementation of the encoder used in *V‑JEPA* (Meta FAIR, 2024) for frozen downstream evaluation.  The design follows the public facebookresearch/jepa reference code but strips it down to the minimum needed for image‑only tasks.

Key implementation choices
--------------------------
1. Most modules are verbatim (or lightly edited for typing/PEP‑8) copies from the upstream repo so that we do *not* depend on the unpublished PyPI package:

2. The official frozen image classification script (https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/evals/image_classification_frozen/eval.py#L451-L455) repeats a still image along the temporal axis before feeding it to the video‑ViT. We reproduce that behaviour with `x = einops.repeat(batch, "b c h w -> b c f h w", f=self.n_frames)` so the model sees a 16‑frame clip of identical images.

3. Checkpoints live at `https://dl.fbaipublicfiles.com/jepa/<ckpt>/<ckpt>.pth.tar`. We download them into an `$CACHE/vjepa` sub‑folder (`download()` helper) to avoid git‑annex or HF dependencies.

4. Only the EMA target encoder (`state["target_encoder"]`) is loaded, mirroring the authors’ evaluation code.  The usual `module.` prefix is stripped so that the state dict matches our local module names.

5. `img_encode()` returns both the full patch grid and a **max‑pooled** global descriptor (`x.max(dim=1).values`).  The attentive classifier used in the paper is *not* re‑implemented here; you can bolt your own head on top of the returned per‑patch features.

## Limitations / divergences from FAIR reference

* No mixed‑precision, distributed training or attentive probe head. This module is encoder‑only.
* Patch/Tubelet shapes are fixed (16x16x2) and `num_frames` is pinned to 16; if you need other variants, expose them through the constructor.
* Only the three public checkpoints (`vitl16`, `vith16`, `vith16-384`) are supported, but extending to future releases is one line in `__init__`.
"""

import functools
import math
import os
import pathlib

import beartype
import einops
import numpy as np
import requests
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import helpers, registry


@beartype.beartype
def get_3d_sincos_pos_embed(
    embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


@beartype.beartype
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(
        grid_w, grid_h
    )  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


@beartype.beartype
def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@beartype.beartype
class PatchEmbed(torch.nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@beartype.beartype
class PatchEmbed3D(torch.nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = torch.nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@beartype.beartype
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@beartype.beartype
class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


@beartype.beartype
class Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias=False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer=torch.nn.GELU,
        norm_layer=torch.nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


@beartype.beartype
class VisionTransformer(torch.nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=torch.nn.LayerNorm,
        init_std: float = 0.02,
        out_layers=None,
        uniform_power=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Position embedding
        self.uniform_power = uniform_power
        self.pos_embed = None
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        # Attention Blocks
        self.blocks = torch.nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=torch.nn.GELU,
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power,
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x):
        """
        :param x: input image/video
        """

        # Tokenize input
        pos_embed = self.pos_embed
        if pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape

        if self.is_video:
            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:
            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2
                ),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_large(patch_size: int = 16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(patch_size: int = 16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@jaxtyped(typechecker=beartype.beartype)
class VJEPA(registry.VisionBackbone):
    def __init__(self, ckpt: str):
        super().__init__()
        self.n_frames = 16
        if ckpt == "vitl16":
            size = 224
            vit = vit_large(img_size=size, num_frames=self.n_frames)
        elif ckpt == "vith16":
            size = 224
            vit = vit_huge(img_size=size, num_frames=self.n_frames)
            size = 224
        elif ckpt == "vith16-384":
            size = 384
            vit = vit_huge(img_size=size, num_frames=self.n_frames)
        else:
            raise ValueError(f"ckpt '{ckpt}' not recognized.")
        self.backbone = vit
        self.size = size

        ckpt_url = f"https://dl.fbaipublicfiles.com/jepa/{ckpt}/{ckpt}.pth.tar"
        state = torch.load(download(ckpt_url), map_location="cpu")
        state = {
            k.replace("module.", ""): v for k, v in state["target_encoder"].items()
        }
        self.load_state_dict(state)

    def img_encode(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> registry.EncodedImgBatch:
        x = einops.repeat(batch, "b c w h -> b c f w h", f=self.n_frames)
        x = self.backbone(x)

        # Reshape to (b, D, N, C) then average over D=8
        depth = self.n_frames // 2  # 8
        x = einops.rearrange(x, "b (d n) c -> b d n c", d=depth)
        x = x.mean(dim=1)

        # Return image features.
        return registry.EncodedImgBatch(x.max(dim=1).values, x)

    def make_img_transform(self):
        import torch
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=self.size),
            v2.CenterCrop(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])


@beartype.beartype
def download(url: str, *, force: bool = False) -> pathlib.Path:
    root = pathlib.Path(helpers.get_cache_dir()) / "vjepa"
    root.mkdir(parents=True, exist_ok=True)
    fname = root / os.path.basename(url)
    if not fname.exists() or force:
        with requests.get(url, stream=True) as r, open(fname, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return fname
