#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from typing import Tuple, Union, Callable, Optional
from functools import partial


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch import Tensor
import torch.nn.functional as F

from copy import deepcopy

#!pip install ipywidgets -q


# # Utils

# In[2]:


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-4,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        assert False, "not implemented yet"
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FlashNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, eps=1e-6, bias=True, rank=None):
        super().__init__(in_features, out_features, bias)
        self.rms_weight = nn.Parameter(torch.ones(in_features))
        self.eps = eps
        _ = rank  # Not used for now

    def forward(self, x: Tensor) -> Tensor:
        # RMS normalize
        ms = (x**2).mean(dim=-1, keepdim=True) + self.eps
        x = x / (ms**0.5)
        # Fuse scaling into weight
        scaled_weight = self.weight * self.rms_weight.unsqueeze(0)
        return F.linear(x, scaled_weight, self.bias)


# In[3]:


class FlashMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Only replace fc1 with FlashNormLinear
        self.fc1 = FlashNormLinear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)  # includes RMSNorm internally
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiMlp(nn.Module):
    def __init__(self, mlp_base, norm_layer=nn.Identity, flash_mlp=True):
        super().__init__()
        self.mlps = nn.ModuleList([deepcopy(mlp_base) for _ in range(1)])
        d = mlp_base.fc1.in_features
        self.flash_mlp = flash_mlp
        self.norms = nn.ModuleList([norm_layer(d) for _ in range(1 * (not flash_mlp))])

    def forward(self, x):
        if self.flash_mlp:
            return self.flash_forward(x)
        with torch.profiler.record_function("MLP Norm C"):
            x = self.norms[0](x)
        with torch.profiler.record_function("MLP C"):
            x = self.mlps[0](x)
        return x

    def flash_forward(self, x):
        with torch.profiler.record_function("FlashMLP C"):
            x = self.mlps[0](x)
        return x


# # LinearContextAttention

# In[4]:


class ClsDepProjOrgAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        hidden_factor=4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.n_h = num_heads
        self.h_d = dim // num_heads

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_factor),
            nn.GELU(),
            nn.Linear(dim * hidden_factor, dim * 7),  # g, q_a, k_a, v_a, q_b, k_b, v_b
        )
        self.qkv_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.attn_drop_v = attn_drop
        self.out_drop = nn.Dropout(proj_drop)

    def scale_shift_reshape(self, x, a, b):
        B, L, d = x.shape
        x_film = a.unsqueeze(1) * x + b.unsqueeze(1)
        return x_film.view(B, L, self.n_h, self.h_d).transpose(1, 2)

    def proj(self, x, cond_out, only_KV=False):
        g, q_a, k_a, v_a, q_b, k_b, v_b = cond_out
        with torch.profiler.record_function("FilmProj QKV: gate & norm"):
            x_gated = x * F.gelu(g).unsqueeze(1)  # [B, S, d]
            x_proj = F.normalize(self.qkv_proj(x_gated), dim=-1, eps=1e-6)  # [B, S, d]

        with torch.profiler.record_function("FilmProj QKV: scale_shift_reshape"):
            Q = None
            if not only_KV:
                Q = self.scale_shift_reshape(x_proj, q_a, q_b)
            K = self.scale_shift_reshape(x_proj, k_a, k_b)
            V = self.scale_shift_reshape(x_proj, v_a, v_b)
        return Q, K, V

    def forward(self, x):
        B, N, C = x.shape
        cls = x[:, 0, :]  # [B, d]

        with torch.profiler.record_function("Film Cond Out: CLS"):
            cond_out = self.mlp(cls).chunk(7, dim=-1)  # 7 × [B, d]

        with torch.profiler.record_function("FilmProj QKV: C & P"):
            q, k, v = self.proj(x, cond_out)

        # 2) scaled‐dot‐product‐attention with dropout
        #    (dropout is only active when self.training==True)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # → [B, H, N, D]

        # 3) re‐assemble heads → [B, N, C]
        attn_out = attn_out.transpose(1, 2).reshape(  # [B, N, H, D]
            B, N, C
        )  # [B, N, H⋅D]

        # 4) output projection + dropout
        x = self.out_proj(attn_out)
        x = self.out_drop(x)
        return x


# # Block
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layerscale=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        flash_mlp=False,
    ) -> None:
        super().__init__()
        self.attn = ClsDepProjOrgAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(layerscale) if layerscale is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp = FlashMlp if flash_mlp else Mlp
        mlp_base = mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )

        self.mlp = MultiMlp(mlp_base, norm_layer=norm_layer, flash_mlp=flash_mlp)
        self.ls2 = LayerScale(layerscale) if layerscale is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(x))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(x))

        def res_con(x, y):
            return x + y

        if self.training and self.sample_drop_ratio > 0.1:
            assert False, "not implemented yet"
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = res_con(x, self.drop_path1(attn_residual_func(x)))
            x = res_con(x, self.drop_path2(ffn_residual_func(x)))
        else:
            x = res_con(x, attn_residual_func(x))
            x = res_con(x, ffn_residual_func(x))
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    assert False, "not implemented yer"
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


# # Patch Embed

# In[7]:


# # Patch emb
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int), x
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 96,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.n_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


# # Vit

# In[ ]:


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module)
    return module


def init_weights_vit_timm(module: nn.Module):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class ClsDepProjOrgAttnVit(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=True,
        layerscale=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        token_drop=0,
        n_registers=0,
        flash_mlp=False,
        return_cls_only=True,
        **kwargs,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            layerscale (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_token_drop = token_drop
        self.n_registers = n_registers
        self.return_cls_only = return_cls_only

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1 + self.n_registers, embed_dim))
        num_pos_emb = 1 + self.n_registers + self.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos_emb, embed_dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                layerscale=layerscale,
                flash_mlp=flash_mlp,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens(self, x):
        with torch.profiler.record_function("Patch Embed"):
            x = self.patch_embed(x)
        with torch.profiler.record_function("prepare Tokens"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1) + self.pos_embed
        with torch.profiler.record_function("Token Drop"):
            x = self.token_drop(x)
        return x

    def token_drop(self, x):
        if not self.p_token_drop or not self.training:
            return x
        patch_idx = 1 + self.n_registers
        non_patches, patches = x[:, :patch_idx, :], x[:, patch_idx:, :]
        num_keep = int((1 - self.p_token_drop) * self.n_patches)
        r = torch.rand(x.size(0), self.n_patches, device=x.device)
        batch_perms = torch.argsort(r, dim=1)[:, :num_keep]  # [B, num_keep]

        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)  # [B, 1]
        patches = patches[batch_idx, batch_perms]  # [B, num_keep, D]
        return torch.cat([non_patches, patches], dim=1)  # [B,1+num_keep,D]

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        with torch.profiler.record_function("Final Norm"):
            out = self.norm(x)
        return out[:, 0, :] if self.return_cls_only else out


# # END