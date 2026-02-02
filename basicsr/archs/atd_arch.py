'''
Official PyTorch implementation of "Structured Feature Interaction with 
Stability Constraints for Image Super-Resolution (SFIS)".

The framework integrates Structure-Preserving Low-Rank Interaction (SLIM) 
and Regularized Dictionary-Induced Feature Transformation (RDFT).
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
import numbers
import os
import cv2
import datetime
from typing import List, Tuple

# ==============================================================================
# Utility Functions
# ==============================================================================

def save_feature_maps(input_tensor, output_dir):
    """
    Visualize and save feature maps for analysis.
    """
    sr_tensor = input_tensor
    [c, h, w] = sr_tensor[0].shape
    merged_map = np.zeros((h, w))

    # Ensure unique directory to prevent overwriting
    dst = output_dir
    if os.path.exists(dst):
        base_name = os.path.basename(os.path.normpath(dst))
        parent_dir = os.path.dirname(dst)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dst = os.path.join(parent_dir, f"{base_name}_{timestamp}")

    os.makedirs(dst, exist_ok=True)
    print(f"[Visualizer] Feature maps will be saved to: {dst}")

    # Process individual channels
    feature_data = np.asarray(sr_tensor[0].data.cpu())
    channel_weights = []

    for i in range(c):
        feat = feature_data[i, :, :]
        # Calculate weight based on mean activation
        channel_weights.append(np.mean(feat))
        
        # Normalize for visualization
        feat_vis = abs(feat)
        feat_vis = (feat_vis - np.min(feat_vis)) / (np.max(feat_vis) + 1e-6)
        feat_vis = np.asarray(feat_vis * 255, dtype=np.uint8)
        feat_vis = cv2.applyColorMap(feat_vis, cv2.COLORMAP_JET)

        # Save upscaled feature map
        save_path = os.path.join(dst, f"channel_{i}_x4.png")
        feat_resized = cv2.resize(feat_vis, (4 * w, 4 * h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path, feat_resized)

    # Weighted fusion
    for j in range(h):
        for k in range(w):
            for i in range(c):
                merged_map[j][k] += channel_weights[i] * feature_data[i][j][k]

    merged_map = abs(merged_map)
    merged_map = (merged_map - np.min(merged_map)) / (np.max(merged_map) + 1e-6)
    merged_map = np.asarray(merged_map * 255, dtype=np.uint8)
    merged_map = cv2.applyColorMap(merged_map, cv2.COLORMAP_JET)

    final_path = os.path.join(dst, 'merged_heatmap_x4.png')
    final_img = cv2.resize(merged_map, (4 * w, 4 * h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(final_path, final_img)
    print(f"[Visualizer] Merged heatmap saved.")


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class DepthWiseConv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(DepthWiseConv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, 
                      padding=(kernel_size - 1) // 2, dilation=1, groups=hidden_features), 
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class LocalFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = DepthWiseConv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class StructuredWindowAttention(nn.Module):
    r"""
    Window-based Self-Attention with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def flops(self, n):
        flops = 0
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


# ==============================================================================
# Core SFIS Modules (SLIM & RDFT)
# ==============================================================================

class RDFT(nn.Module):
    """
    Regularized Dictionary-Induced Feature Transformation (RDFT).
    Originally implemented as dictionary-based cross attention.
    """
    def __init__(self, dim, input_resolution, num_tokens=64, reducted_dim=16, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.rc = reducted_dim

        # Projections for dictionary interaction
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = nn.LayerNorm(reducted_dim)
        self.k_norm = nn.LayerNorm(reducted_dim)

        # Dictionary refinement (Spatial consistency)
        self.v_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU()
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, dict_tokens, x_size):
        """
        x: (B, N, C) - Input features
        dict_tokens: (B, M, C) - Learned dictionary priors
        """
        B, N, C = x.shape
        _, M, _ = dict_tokens.shape
        H, W = x_size

        # Embed into low-rank space for stable matching
        q = self.q_norm(self.wq(x))
        k = self.k_norm(self.wk(dict_tokens))

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Regularized Correspondence
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / torch.clamp(self.temperature, min=0.5)
        attn = self.softmax(attn)

        # Value transformation with spatial priors
        v = self.wv(dict_tokens)
        v_img = v.transpose(1, 2).unsqueeze(-1)  # (B, C, M, 1)
        v_img = self.v_refine(v_img)
        v = v_img.squeeze(-1).transpose(1, 2)

        out = torch.matmul(attn, v)
        return out, attn

    def flops(self, n):
        # Approximate FLOPs
        flops = n * self.dim * self.rc * 2 
        flops += n * self.num_tokens * self.rc
        return flops


class ReGroupContiguous(nn.Module):
    """
    Helper for SLIM: Efficient channel grouping.
    """
    def __init__(self, groups: List[int] = [1, 1, 2, 4], sort_by_energy: bool = False):
        super(ReGroupContiguous, self).__init__()
        self.groups = groups
        self.sort_by_energy = sort_by_energy

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, C, N = q.shape
        if self.sort_by_energy and N > 1:
            energy = q.abs().mean(dim=-1).mean(dim=0)
            _, sorted_idx = torch.sort(energy, descending=True)
        else:
            sorted_idx = torch.arange(C, device=q.device)

        total = sum(self.groups)
        group_sizes = [int(r * C / total) for r in self.groups]
        s = sum(group_sizes)
        if s < C:
            group_sizes[-1] += (C - s)
        elif s > C:
            group_sizes[-1] -= (s - C)

        q_sorted = q[:, sorted_idx, :]
        k_sorted = k[:, sorted_idx, :]
        v_sorted = v[:, sorted_idx, :]

        q_groups, k_groups, v_groups, idx_groups = [], [], [], []
        start = 0
        for g in group_sizes:
            end = start + g
            idx = sorted_idx[start:end]
            idx_groups.append(idx)
            q_groups.append(q_sorted[:, start:end, :])
            k_groups.append(k_sorted[:, start:end, :])
            v_groups.append(v_sorted[:, start:end, :])
            start = end

        return q_groups, k_groups, v_groups, idx_groups


class GroupLowRankAttention(nn.Module):
    """
    Component of SLIM: Low-rank group interaction.
    Compresses channel dimension Cg -> r to model dependencies efficiently.
    """
    def __init__(self, Cg: int, r: int = None):
        super(GroupLowRankAttention, self).__init__()
        self.Cg = Cg
        if r is None:
            r = max(8, Cg // 4)
        self.r = min(r, Cg)
        
        # Low-rank projections (Cg -> r)
        self.q_proj = nn.Conv1d(Cg, self.r, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(Cg, self.r, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(Cg, self.r, kernel_size=1, bias=False)
        self.back_proj = nn.Conv1d(self.r, Cg, kernel_size=1, bias=False)

    def forward(self, qg, kg, vg, temp):
        # qg, kg, vg: (B, Cg, N)
        Qr = self.q_proj(qg)
        Kr = self.k_proj(kg)
        Vr = self.v_proj(vg)

        Qr_n = F.normalize(Qr, dim=-1)
        Kr_n = F.normalize(Kr, dim=-1)

        # Structure-preserving interaction in low-rank space
        att = torch.matmul(Qr_n, Kr_n.transpose(-2, -1)) * temp
        att = F.softmax(att, dim=-1)

        out_r = torch.matmul(att, Vr)
        out_g = self.back_proj(out_r)
        return out_g


class SpatialContextExtractor(nn.Module):
    def __init__(self, channels: int, scales: List[int] = [1, 2]):
        super(SpatialContextExtractor, self).__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scales = scales

    def forward(self, v, idx_groups):
        x = self.dw(v)
        x = F.gelu(self.pw(x))
        B, C, H, W = x.shape
        ctxs = []
        for idx in idx_groups:
            xg = x[:, idx, :, :]
            pooled = []
            for s in self.scales:
                if s == 1:
                    pg = F.adaptive_avg_pool2d(xg, (1, 1)).squeeze(-1).squeeze(-1)
                else:
                    ph = max(1, H // s)
                    pw_ = max(1, W // s)
                    pg = F.adaptive_avg_pool2d(xg, (ph, pw_)).reshape(B, xg.shape[1], -1).mean(dim=-1)
                pooled.append(pg)
            ctx = torch.stack(pooled, dim=0).mean(dim=0)
            ctxs.append(ctx)
        return ctxs


class InterGroupModulator(nn.Module):
    def __init__(self, Cg: int, reduction: int = 4):
        super(InterGroupModulator, self).__init__()
        hidden = max(1, Cg // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(Cg, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * Cg)
        )

    def forward(self, ctx):
        out = self.mlp(ctx)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class IntraGroupModulator(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super(IntraGroupModulator, self).__init__()
        hidden = max(1, channels // reduction)
        self.down = nn.Conv1d(channels, hidden, kernel_size=1)
        self.up = nn.Conv1d(hidden, channels, kernel_size=1)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, ref):
        fused = x + ref
        g = F.gelu(self.gate(fused)) * fused
        p = self.up(self.down(g))
        return p


class SLIM(nn.Module):
    """
    Structure-Preserving Low-Rank Interaction Module (SLIM).
    Decomposes interactions into grouped low-rank components with stability constraints.
    """
    def __init__(self, dim: int, groups: List[int] = [1, 1, 2, 4], bias: bool = False, low_rank_r: int = None):
        super(SLIM, self).__init__()
        self.dim = dim
        self.groups = groups
        self.num_groups = len(groups)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Feature Grouping and Low-Rank logic
        self.regroup = ReGroupContiguous(groups=groups, sort_by_energy=False)
        total = sum(groups)
        self.group_sizes = [int(g * dim / total) for g in groups]
        s = sum(self.group_sizes)
        if s < dim:
            self.group_sizes[-1] += (dim - s)

        # Low-rank group attentions
        self.group_attns = nn.ModuleList([GroupLowRankAttention(gsize, r=low_rank_r) for gsize in self.group_sizes])

        # Spatial context for stability
        self.ctx_extractor = SpatialContextExtractor(dim, scales=[1, 2])
        self.inter_mods = nn.ModuleList([InterGroupModulator(gsize) for gsize in self.group_sizes])
        self.intra = IntraGroupModulator(channels=dim, reduction=4)

        self.temperature = nn.Parameter(torch.ones(self.num_groups, 1, 1))
        self.res_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v_flat = rearrange(v, 'b c h w -> b c (h w)')

        # Decompose into subspaces
        q_groups, k_groups, v_groups, idx_groups = self.regroup(q, k, v_flat)
        ctx_list = self.ctx_extractor(v, idx_groups)

        out_groups = []
        tmp_groups = []

        # Process each subspace
        for i, (qg, kg, vg, ctx, group_attn, inter_mod) in enumerate(
                zip(q_groups, k_groups, v_groups, ctx_list, self.group_attns, self.inter_mods)):
            
            # Low-rank interaction
            out_g = group_attn(qg, kg, vg, temp=self.temperature[i])

            # Inter-channel stability modulation
            scale, shift = inter_mod(ctx)
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
            out_g = out_g * (1.0 + scale) + shift

            out_groups.append(out_g)
            tmp_groups.append((qg.detach() + kg.detach()))

        out_all = torch.cat(out_groups, dim=1)
        tmp_all = torch.cat(tmp_groups, dim=1)

        # Intra-channel stability modulation
        intra_delta = self.intra(out_all, tmp_all)
        out_all = out_all + intra_delta

        out_all = rearrange(out_all, 'b c (h w) -> b c h w', h=H, w=W)
        out_proj = self.project_out(out_all)

        return x + self.res_alpha * out_proj


# ==============================================================================
# SFIS Layer and Block
# ==============================================================================

class SFISLayer(nn.Module):
    r"""
    Basic Layer of SFIS.
    Combines Window Attention and RDFT (Regularized Dictionary Transform).
    """
    def __init__(self, dim, idx, input_resolution, num_heads, window_size, shift_size,
                 category_size, num_tokens, reducted_dim, convffn_kernel_size,
                 mlp_ratio, qkv_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_tokens = num_tokens
        self.is_last = is_last

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Dictionary Update Mechanics
        if not is_last:
            self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True)
            self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)

        # Standard Window Attention (Local)
        self.attn_win = StructuredWindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias
        )
        
        # RDFT: Dictionary-Induced Interaction (Global/Priors)
        self.rdft = RDFT(
            self.dim, input_resolution=input_resolution, qkv_bias=qkv_bias,
            num_tokens=num_tokens, reducted_dim=reducted_dim
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = LocalFFN(in_features=dim, hidden_features=mlp_hidden_dim, 
                            kernel_size=convffn_kernel_size, act_layer=act_layer)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, dict_features, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # Apply RDFT (Dictionary Interaction)
        x_rdft, sim_map = self.rdft(x, dict_features, x_size)

        # SW-MSA Logic
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None

        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)

        attn_windows = self.attn_win(x_windows, rpi=params['rpi_sa'], mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x_win = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_win = shifted_x
        
        # Combine Local Window + Global RDFT
        x = shortcut + x_win.view(b, n, c) + x_rdft

        # FFN
        x = x + self.ffn(self.norm2(x), x_size)

        # Adaptive Dictionary Refinement
        if not self.is_last:
            mask_soft = self.softmax(self.norm3(sim_map.transpose(-1, -2)))
            mask_x = x.reshape(b, n, c)
            s = self.sigmoid(self.sigma)
            dict_features = s * dict_features + (1-s) * torch.einsum('btn,bnc->btc', mask_soft, mask_x)

        return x, dict_features

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += self.dim * 3 * self.dim * h * w
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.rdft.flops(h * w)
        flops += 2 * h * w * self.dim * self.dim * 2 # Approximated MLP
        return flops


class SFISBlock(nn.Module):
    """ 
    Core Block of the SFIS Network. 
    Integrates multiple SFIS Layers and manages the shared Dictionary Priors.
    """
    def __init__(self, dim, idx, input_resolution, depth, num_heads, window_size, category_size,
                 num_tokens, convffn_kernel_size, reducted_dim, mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SFISLayer(
                    dim=dim, idx=i, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                    category_size=category_size, num_tokens=num_tokens, convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    norm_layer=norm_layer, is_last=i == depth-1
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Learned Dictionary Priors
        self.dictionary_priors = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        # Expand dictionary for batch
        dict_features = self.dictionary_priors.repeat([b, 1, 1])
        
        for layer in self.layers:
            if self.use_checkpoint:
                layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x, dict_features = layer(x, dict_features, x_size, params)
            
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        return flops


# ==============================================================================
# Main SFIS Network Architecture
# ==============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    def flops(self, input_resolution=None):
        h, w = self.img_size if input_resolution is None else input_resolution
        return h * w * self.embed_dim

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x
    
    def flops(self, input_resolution=None):
        return 0


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        return h * w * self.num_feat * 3 * 9


@ARCH_REGISTRY.register()
class SFIS(nn.Module):
    r""" 
    SFIS: Structured Feature Interaction with Stability Constraints.
    """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 category_size=128,
                 num_tokens=64,
                 reducted_dim=4,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.register_buffer('relative_position_index_SA', self.calculate_rpi_sa())

        # Build SFIS Blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SFISBlock(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                category_size=category_size,
                num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(embed_dim)

        # SLIM: Structure-Preserving Low-Rank Interaction (Global Enhancement)
        self.slim_module = SLIM(dim=embed_dim, groups=[1, 1, 2, 4], low_rank_r=8)

        # Residual connection handling
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # Reconstruction
        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (self.patch_embed.patches_resolution[0], self.patch_embed.patches_resolution[1]))
        else:
             self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        # Processing through SFIS blocks (RDFT inside)
        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # 1. First Convolution
        x_first = self.conv_first(x)
        
        # 2. Main Body (SFIS Blocks)
        res = self.forward_features(x_first, params)
        
        # 3. Global Structural Refinement (SLIM)
        # Applying SLIM here to refine the features globally before upsampling
        res_slim = self.slim_module(res)

        # 4. Residual Connection
        x_body = self.conv_after_body(res_slim) + x_first

        # 5. Upsampling
        if self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x_body)
        else:
            x = self.conv_last(x_body)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.patch_embed.patches_resolution if input_resolution is None else input_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops((h, w))
        for layer in self.layers:
            flops += layer.flops((h, w))
        flops += self.slim_module.count_parameters() # Approx
        if self.upsampler == 'pixelshuffledirect':
            flops += self.upsample.flops((h, w))
        return flops

if __name__ == '__main__':
    upscale = 4
    model = SFIS(
        upscale=4,
        img_size=64,
        embed_dim=48,
        depths=[6, 6, 6, 6],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        category_size=128,
        num_tokens=64,
        reducted_dim=8,
        convffn_kernel_size=7,
        img_range=1.,
        mlp_ratio=1,
        upsampler='pixelshuffledirect')

    total = sum([param.nelement() for param in model.parameters()])
    print("SFIS Model created.")
    print("Number of parameters: %.3fM" % (total / 1e6))
    
    _input = torch.randn([2, 3, 64, 64])
    output = model(_input)
    print("Output Shape:", output.shape)