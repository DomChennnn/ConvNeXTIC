# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .gdn import GDN
from .natten import NeighborhoodAttention

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "CheckerboardMaskedConv2d",
    "MultistageMaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    "QReLU",
    "RSTB",
    "ResViTBlock",
]


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class CheckerboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x: Tensor) -> Tensor:
        # self.mask[:, :, :, :] = 0
        # self.mask[:, :, 0:1, 1:2] = 1
        # print(self.mask)
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MultistageMaskedConv2d_v2(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2 ** bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
                torch.exp(
                    (-ctx.alpha ** ctx.beta)
                    * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
                )
                * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint
                                         )

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, x_size):
        return self.patch_unembed(self.residual_group(self.patch_embed(x), x_size), x_size) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class NSABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicViTLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NSABlock(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ResViTBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super(ResViTBlock, self).__init__()
        self.dim = dim

        self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=drop_path_rate, norm_layer=norm_layer)

    def forward(self, x):
        return self.residual_group(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x

# class GaussianFilter(nn.Module):
#     def __init__(self, kernel_size=13, stride=1, padding=6):
#         super(GaussianFilter, self).__init__()
#         # initialize guassian kernel
#         mean = (kernel_size - 1) / 2.0
#         variance = ((kernel_size - 1) / 6.0) ** 2.0
#         # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#         x_coord = torch.arange(kernel_size)
#         x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#         y_grid = x_grid.t()
#         xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

#         # Calculate the 2-dimensional gaussian kernel
#         gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

#         # Make sure sum of values in gaussian kernel equals 1.
#         gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

#         # Reshape to 2d depthwise convolutional weight
#         gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
#         gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

#         # create gaussian filter as convolutional layer
#         self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
#         self.gaussian_filter.weight.data = gaussian_kernel
#         self.gaussian_filter.weight.requires_grad = False

#     def forward(self, x):
#         return self.gaussian_filter(x)


# class FilterLow(nn.Module):
#     def __init__(self, recursions=1, kernel_size=9, stride=1, padding=True, include_pad=True, gaussian=False):
#         super(FilterLow, self).__init__()
#         if padding:
#             pad = int((kernel_size - 1) / 2)
#         else:
#             pad = 0
#         if gaussian:
#             self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
#         else:
#             self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad,
#                                        count_include_pad=include_pad)
#         self.recursions = recursions

#     def forward(self, img):
#         for i in range(self.recursions):
#             img = self.filter(img)
#         return img


# class FilterHigh(nn.Module):
#     def __init__(self, recursions=1, kernel_size=9, stride=1, include_pad=True, normalize=True, gaussian=False):
#         super(FilterHigh, self).__init__()
#         self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
#                                     gaussian=gaussian)
#         self.recursions = recursions
#         self.normalize = normalize

#     def forward(self, img):
#         if self.recursions > 1:
#             for i in range(self.recursions - 1):
#                 img = self.filter_low(img)
#         img = img - self.filter_low(img)
#         if self.normalize:
#             return 0.5 + img * 0.5
#         else:
#             return img


# class SE(nn.Module):
#     def __init__(self, inp, oup, expansion=0.25):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(oup, int(inp * expansion)),
#             nn.GELU(),
#             nn.Linear(int(inp * expansion), oup),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class ConvMixer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             SE(dim, dim),
#             nn.Conv2d(dim, dim, 1, 1, 0),
#         )

#     def forward(self, x):
#         x = x.contiguous()
#         x = x.permute(0, 3, 1, 2)
#         x = x + self.conv(x)
#         x = x.permute(0, 2, 3, 1)
#         return x


# class NSABlock(nn.Module):
#     def __init__(self, dim, num_heads, kernel_size=7,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio

#         self.norm1 = norm_layer(dim)
#         self.attn = NeighborhoodAttention(
#             dim, kernel_size=kernel_size, num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.layer_scale = False
#         if layer_scale is not None and type(layer_scale) in [int, float]:
#             self.layer_scale = True
#             self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
#             self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
#         else:
#             self.gamma1 = 1.0
#             self.gamma2 = 1.0

#     def forward(self, x):
#         shortcut = x
#         x = self.norm1(x)
#         x = self.attn(x)
#         x = shortcut + self.drop_path(self.gamma1 * x)
#         x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
#         return x


# class BasicViTLayer(nn.Module):
#     def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4.,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm,
#                  layer_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.depth = depth

#         self.blocks = nn.ModuleList([
#             NSABlock(dim=dim,
#                      num_heads=num_heads, kernel_size=kernel_size,
#                      mlp_ratio=mlp_ratio,
#                      qkv_bias=qkv_bias, qk_scale=qk_scale,
#                      drop=drop, attn_drop=attn_drop,
#                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                      norm_layer=norm_layer, layer_scale=layer_scale)
#             for i in range(depth)])

#     def forward(self, x):
#         for blk in self.blocks:
#             x = blk(x)
#         return x


# class ResViTBlock(nn.Module):
#     def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
#                  qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0.2, norm_layer=nn.LayerNorm, layer_scale=None):
#         super(ResViTBlock, self).__init__()
#         self.dim = dim

#         self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
#                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
#                                          drop_path=drop_path_rate, norm_layer=norm_layer, layer_scale=layer_scale)

#     def forward(self, x):
#         return self.residual_group(x.permute(0,2,3,1)).permute(0,3,1,2) + x


# def grid_partition(
#         input: torch.Tensor,
#         grid_size: Tuple[int, int] = (7, 7)
# ) -> torch.Tensor:
#     """ Grid partition function.
#     Args:
#         input (torch.Tensor): Input tensor of the shape [B, C, H, W].
#         grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)
#     Returns:
#         grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
#     """
#     # Get size of input
#     B, C, H, W = input.shape
#     # Unfold input
#     grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
#     # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
#     grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
#     return grid


# def grid_reverse(
#         grid: torch.Tensor,
#         original_size: Tuple[int, int],
#         grid_size: Tuple[int, int] = (7, 7)
# ) -> torch.Tensor:
#     """ Reverses the grid partition.
#     Args:
#         Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
#         original_size (Tuple[int, int]): Original shape.
#         grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)
#     Returns:
#         output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
#     """
#     # Get height, width, and channels
#     (H, W), C = original_size, grid.shape[-1]
#     # Compute original batch size
#     B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
#     # Fold grid tensor
#     output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
#     output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
#     return output


# class CausalAttentionModule(nn.Module):
#     r""" Causal multi-head self attention module.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#     """
#     def __init__(self, dim, out_dim, window_size=4, num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.window_size = window_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.attn_drop = nn.Dropout(attn_drop)

#         self.norm1 = nn.LayerNorm(dim)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.mask = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(1, self.block_size, 1)

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * block_len - 1) * (2 * block_len - 1), num_heads))  # 2*P-1 * 2*P-1, num_heads

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(block_len)
#         coords_w = torch.arange(block_len)
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, P, P
#         coords_flatten = torch.flatten(coords, 1)  # 2, P*P
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, PP, PP
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # PP, PP, 2
#         relative_coords[:, :, 0] += block_len - 1  # shift to start from 0
#         relative_coords[:, :, 1] += block_len - 1
#         relative_coords[:, :, 0] *= 2 * block_len - 1
#         relative_position_index = relative_coords.sum(-1)  # PP, PP
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.softmax = nn.Softmax(dim=-1)

#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=attn_drop)
#         self.proj = nn.Linear(dim, out_dim)

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         B, H, W, C = x.shape
#         x_window = window_partition(x, self.window_size)
#         x_window = x_window.view(-1, self.window_size**2, C) # BHW, PP, C


#         x_unfold = F.unfold(x, kernel_size=(5, 5), padding=2) # B, CPP, HW
#         x_unfold = x_unfold.reshape(B, C, self.block_size, H*W).permute(0, 3, 2, 1).contiguous().view(-1, self.block_size, C) # BHW, PP, C

#         x_masked = x_unfold * self.mask.to(x_unfold.device)
#         out = self.norm1(x_masked)
#         qkv = self.qkv(out).reshape(B*H*W, self.block_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, BHW, num_heads, PP, C
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) # BHW, num_heads, PP, C//num_heads
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1)) # BHW, num_heads, PP, PP

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.block_size, self.block_size, -1)  # PP, PP, num_heads
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, PP, PP
#         attn = attn + relative_position_bias.unsqueeze(0)

#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         out = (attn @ v).transpose(1, 2).reshape(B*H*W, self.block_size, C) # [BHW, num_heads, PP, PP] [BHW, num_heads, PP, C//num_heads]
#         out += x_masked
#         out_sumed = torch.sum(out, dim=1).reshape(B, H*W, C)
#         out = self.norm2(out_sumed)
#         out = self.mlp(out)
#         out += out_sumed

#         out = self.proj(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2) # B, C_out, H, W

#         return out


# def _to_channel_last(x):
#     """
#     Args:
#         x: (B, C, H, W)
#     Returns:
#         x: (B, H, W, C)
#     """
#     return x.permute(0, 2, 3, 1)


# def _to_channel_first(x):
#     """
#     Args:
#         x: (B, H, W, C)
#     Returns:
#         x: (B, C, H, W)
#     """
#     return x.permute(0, 3, 1, 2)


# class SE(nn.Module):
#     """
#     Squeeze and excitation block
#     """
#     def __init__(self,
#                  inp,
#                  oup,
#                  expansion=0.25):
#         """
#         Args:
#             inp: input features dimension.
#             oup: output features dimension.
#             expansion: expansion ratio.
#         """

#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(oup, int(inp * expansion), bias=False),
#             nn.GELU(),
#             nn.Linear(int(inp * expansion), oup, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class ReduceSize(nn.Module):
#     """
#     Down-sampling block based on: "Hatamizadeh et al.,
#     Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
#     """
#     def __init__(self,
#                  dim,
#                  norm_layer=nn.LayerNorm
#                  ):
#         """
#         Args:
#             dim: feature size dimension.
#             norm_layer: normalization layer.
#             keep_dim: bool argument for maintaining the resolution.
#         """

#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1,
#                       groups=dim),
#             nn.GELU(),
#             SE(dim, dim),
#             nn.Conv2d(dim, dim, 1, 1, 0),
#         )
#         self.reduction = nn.Conv2d(dim, dim, 3, 2, 1)
#         self.norm = norm_layer(dim)

#     def forward(self, x):
#         x = x.contiguous()
#         x = self.norm(x)
#         x = _to_channel_first(x)
#         x = x + self.conv(x)
#         x = self.reduction(x)
#         # x = _to_channel_last(x)
#         return x


# class IncreaseSize(nn.Module):
#     """
#     Down-sampling block based on: "Hatamizadeh et al.,
#     Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
#     """
#     def __init__(self,
#                  dim,
#                  norm_layer=nn.LayerNorm
#                  ):
#         """
#         Args:
#             dim: feature size dimension.
#             norm_layer: normalization layer.
#             keep_dim: bool argument for maintaining the resolution.
#         """

#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1,
#                       groups=dim),
#             nn.GELU(),
#             SE(dim, dim),
#             nn.Conv2d(dim, dim, 1, 1, 0),
#         )
#         self.increase = deconv(dim, dim, kernel_size=3, stride=2)
#         self.norm = norm_layer(dim)

#     def forward(self, x):
#         x = x.contiguous()
#         x = self.norm(x)
#         x = _to_channel_first(x)
#         x = x + self.conv(x)
#         x = self.increase(x)
#         # x = _to_channel_last(x)
#         return x


# class PatchEmbed(nn.Module):
#     """
#     Patch embedding block based on: "Hatamizadeh et al.,
#     Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
#     """
#     def __init__(self, in_chans=3, dim=96):
#         """
#         Args:
#             in_chans: number of input channels.
#             dim: feature size dimension.
#         """

#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, dim, 3, 2, 1)
#         self.conv_down = ReduceSize(dim=dim)

#     def forward(self, x):
#         x = self.proj(x)
#         x = _to_channel_last(x)
#         x = self.conv_down(x)
#         return x


# class PatchUnEmbed(nn.Module):
#     """
#     Patch embedding block based on: "Hatamizadeh et al.,
#     Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
#     """
#     def __init__(self, out_chans=3, dim=192):
#         """
#         Args:
#             in_chans: number of input channels.
#             dim: feature size dimension.
#         """

#         super().__init__()
#         self.conv_up = IncreaseSize(dim=dim)
#         self.proj = deconv(dim, out_chans, kernel_size=3, stride=2)

#     def forward(self, x):
#         x = _to_channel_last(x)
#         x = self.conv_up(x)
#         x = self.proj(x)
#         return x


