import math
import numpy as np
import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import ResViTBlock, CheckerboardMaskedConv2d
from timm.models.layers import trunc_normal_

from .utils import conv, deconv, update_registered_buffers, quantize_ste, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TinyLIC(nn.Module):
    """
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=320):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        num_heads = [8, 12, 16, 20, 12, 12]
        kernel_size = 7
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        self.num_slices = 5
        self.M = M

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(dim=N,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
        )
        self.g_a2 = conv(N, N*3//2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(dim=N*3//2,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
        )
        self.g_a4 = conv(N*3//2, N*2, kernel_size=3, stride=2)
        self.g_a5 = ResViTBlock(dim=N*2,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        self.g_a6 = conv(N*2, M, kernel_size=3, stride=2)
        self.g_a7 = ResViTBlock(dim=M,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )

        self.h_a0 = conv(M, N*3//2, kernel_size=3, stride=2)
        self.h_a1 = ResViTBlock(dim=N*3//2,
                         depth=depths[4],
                         num_heads=num_heads[4],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
        )
        self.h_a2 = conv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_a3 = ResViTBlock(dim=N*3//2,
                         depth=depths[5],
                         num_heads=num_heads[5],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = ResViTBlock(dim=N*3//2,
                         depth=depths[0],
                         num_heads=num_heads[0],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
        )
        self.h_s1 = deconv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_s2 = ResViTBlock(dim=N*3//2,
                         depth=depths[1],
                         num_heads=num_heads[1],
                         kernel_size=kernel_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
        )
        self.h_s3 = deconv(N*3//2, M*2, kernel_size=3, stride=2)
        
        self.g_s0 = ResViTBlock(dim=M,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        self.g_s1 = deconv(M, N*2, kernel_size=3, stride=2)
        self.g_s2 = ResViTBlock(dim=N*2,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )
        self.g_s3 = deconv(N*2, N*3//2, kernel_size=3, stride=2)
        self.g_s4 = ResViTBlock(dim=N*3//2,
                        depth=depths[4],
                        num_heads=num_heads[4],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
        )
        self.g_s5 = deconv(N*3//2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(dim=N,
                        depth=depths[5],
                        num_heads=num_heads[5],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N*3//2)
        self.gaussian_conditional = GaussianConditional(None)

        in_ch_list = [0, 16, 32, 64, 128]
        out_ch_list = [16, 16, 32, 64, M-128]
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(2*M + in_ch_list[i], 224, stride=1, kernel_size=5),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=5),
                nn.GELU(),
                conv(128, 2*out_ch_list[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.sc_transforms = nn.ModuleList(
            CheckerboardMaskedConv2d(
                out_ch_list[i], 2*out_ch_list[i], kernel_size=5, padding=2, stride=1
            ) for i in range(self.num_slices)
        )
        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                conv(2*M + 12//3*out_ch_list[i], 10//3*out_ch_list[i], 1, 1),
                nn.GELU(),
                conv(10//3*out_ch_list[i], 8//3*out_ch_list[i], 1, 1),
                nn.GELU(),
                conv(8//3*out_ch_list[i], 6//3*out_ch_list[i], 1, 1),
            ) for i in range(self.num_slices)
        )

        self.apply(self._init_weights)  

    def g_a(self, x):
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x)
        x = self.g_a6(x)
        x = self.g_a7(x)
        return x

    def g_s(self, x):
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a1(x)
        x = self.h_a2(x)
        x = self.h_a3(x)
        return x

    def h_s(self, x):
        x = self.h_s0(x)
        x = self.h_s1(x)
        x = self.h_s2(x)
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset
        
        params = self.h_s(z_hat)
        
        y_slices = y.split([16, 16, 32, 64, self.M-128], 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

            y_half = y_hat_slice.clone()
            y_half[:, :, 0::2, 0::2] = 0
            y_half[:, :, 1::2, 1::2] = 0

            sc_params = self.sc_transforms[slice_index](y_half)
            sc_params[:, :, 0::2, 1::2] = 0
            sc_params[:, :, 1::2, 0::2] = 0
            
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
            y_hat_slices.append(y_hat_slice)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
            y_likelihood.append(y_slice_likelihood)
       
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # Generate the image reconstruction.
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        y_slices = y.split([16, 16, 32, 64, self.M-128], 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)

            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            scales_hat_anchor, _ = Demultiplexer(scales_hat)
            means_hat_anchor, _ = Demultiplexer(means_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
            y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

            symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
            indexes_list.extend(index_anchor.reshape(-1).tolist())

            y_half = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))

            sc_params = self.sc_transforms[slice_index](y_half)
            sc_params[:, :, 0::2, 1::2] = 0
            sc_params[:, :, 1::2, 0::2] = 0
            
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
            y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

            symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
            indexes_list.extend(index_non_anchor.reshape(-1).tolist())

            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:]
                }

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = torch.cat([params] + y_hat_slices, dim=1)
            cc_params = self.cc_transforms[slice_index](support_slices)

            sc_params = torch.zeros_like(cc_params)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            scales_hat_anchor, _ = Demultiplexer(scales_hat)
            means_hat_anchor, _ = Demultiplexer(means_hat)
            index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
            rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
            y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

            y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
            sc_params = self.sc_transforms[slice_index](y_hat_slice)
            gaussian_params = self.entropy_parameters[slice_index](
                torch.cat((params, sc_params, cc_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            _, scales_hat_non_anchor = Demultiplexer(scales_hat)
            _, means_hat_non_anchor = Demultiplexer(means_hat)
            index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
            rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)
            
            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}






