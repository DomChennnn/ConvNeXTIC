# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from compressai.ans import BufferedRansEncoder, RansDecoder
# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# from compressai.layers import MaskedConv2d
# from timm.models.layers import trunc_normal_
# from .layers import ConvNeXtLayer, MultistageMaskedConv2d
# from .utils import update_registered_buffers, Demultiplexer, Multiplexer
#
# # From Balle's tensorflow compression examples
# SCALES_MIN = 0.11
# SCALES_MAX = 256
# SCALES_LEVELS = 64
#
#
# def get_scale_table(
#         min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
# ):  # pylint: disable=W0622
#     return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
#
#
# class cnexTIC(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         in_chans = config['in_chans']
#         embed_dim = config['embed_dim']
#         latent_dim = config['latent_dim']
#         drop_path_rate = 0.1
#         N = embed_dim
#         M = latent_dim
#
#         depths = [2, 4, 6, 2, 2, 2]
#         # depths = [3, 3, 9, 3, 3, 3]
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         self.g_a0 = ConvNeXtLayer(dim_in=in_chans,
#                                   dim_out=embed_dim,
#                                   depth=depths[0],
#                                   drop = dpr[sum(depths[:0]):sum(depths[:1])],
#                                   is_first = True,
#                                   encode=True)
#         self.g_a1 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[1],
#                                   drop = dpr[sum(depths[:1]):sum(depths[:2])],
#                                   is_first = False,
#                                   encode=True)
#         self.g_a2 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[2],
#                                   drop = dpr[sum(depths[:2]):sum(depths[:3])],
#                                   is_first = False,
#                                   encode=True)
#         self.g_a3 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=latent_dim,
#                                   depth=depths[3],
#                                   drop = dpr[sum(depths[:3]):sum(depths[:4])],
#                                   is_first = False,
#                                   encode=True)
#         self.h_a0 = ConvNeXtLayer(dim_in=latent_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[4],
#                                   drop = dpr[sum(depths[:4]):sum(depths[:5])],
#                                   is_first = False,
#                                   encode=True,
#                                   is_hyper=True)
#         self.h_a1 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[5],
#                                   drop = dpr[sum(depths[:5]):sum(depths[:6])],
#                                   is_first = False,
#                                   encode=True,
#                                   is_hyper=True)
#
#         depths = depths[::-1]
#         self.h_s0 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[0],
#                                   drop = dpr[sum(depths[:0]):sum(depths[:1])],
#                                   is_first = False,
#                                   encode=False,
#                                   is_hyper=True)
#         self.h_s1 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=2*latent_dim,
#                                   depth=depths[1],
#                                   drop = dpr[sum(depths[:1]):sum(depths[:2])],
#                                   is_first = False,
#                                   encode=False,
#                                   is_hyper=True)
#
#         self.g_s0 = ConvNeXtLayer(dim_in=latent_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[2],
#                                   drop = dpr[sum(depths[:2]):sum(depths[:3])],
#                                   is_first = False,
#                                   encode=False)
#         self.g_s1 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[3],
#                                   drop = dpr[sum(depths[:3]):sum(depths[:4])],
#                                   is_first = False,
#                                   encode=False)
#         self.g_s2 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=embed_dim,
#                                   depth=depths[4],
#                                   drop = dpr[sum(depths[:4]):sum(depths[:5])],
#                                   is_first = False,
#                                   encode=False)
#         self.g_s3 = ConvNeXtLayer(dim_in=embed_dim,
#                                   dim_out=in_chans,
#                                   depth=depths[5],
#                                   drop = dpr[sum(depths[:5]):sum(depths[:6])],
#                                   is_first = True,
#                                   encode=False)
#
#         self.entropy_bottleneck = EntropyBottleneck(embed_dim)
#         self.gaussian_conditional = GaussianConditional(None)
#         self.context_prediction_1 = MultistageMaskedConv2d(
#             M, 2*M, kernel_size=3, padding=1, stride=1, mask_type='A'
#         )
#         self.context_prediction_2 = MultistageMaskedConv2d(
#             M, 2*M, kernel_size=3, padding=1, stride=1, mask_type='B'
#         )
#         self.context_prediction_3 = MultistageMaskedConv2d(
#             M, 2*M, kernel_size=3, padding=1, stride=1, mask_type='C'
#         )
#
#         self.entropy_parameters = nn.Sequential(
#             nn.Conv2d(M * 24 // 3, M * 18 // 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(M * 18 // 3, M * 12 // 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(M * 12 // 3, M * 6 // 3, 1, 1),
#         )
#
#         self.apply(self._init_weights)
#
#         # g_a, encoder
#
#     def g_a(self, x):
#         x = self.g_a0(x)
#         x = self.g_a1(x)
#         x = self.g_a2(x)
#         x = self.g_a3(x)
#         return x
#
#     def g_s(self, x):
#         x = self.g_s0(x)
#         x = self.g_s1(x)
#         x = self.g_s2(x)
#         x = self.g_s3(x)
#         return x
#
#     def h_a(self, x):
#         x = self.h_a0(x)
#         x = self.h_a1(x)
#         return x
#
#     def h_s(self, x):
#         x = self.h_s0(x)
#         x = self.h_s1(x)
#         return x
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def forward(self, x):
#         y = self.g_a(x)
#         z = self.h_a(y)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         params = self.h_s(z_hat)
#
#         y_hat = self.gaussian_conditional.quantize(
#             y, "noise" if self.training else "dequantize"
#         )
#
#         y_1 = y_hat.clone()
#         y_1[:, :, 0::2, 1::2] = 0
#         y_1[:, :, 1::2, :] = 0
#         ctx_params_1 = self.context_prediction_1(y_1)
#         ctx_params_1[:, :, 0::2, :] = 0
#         ctx_params_1[:, :, 1::2, 0::2] = 0
#
#         y_2 = y_hat.clone()
#         y_2[:, :, 0::2, 1::2] = 0
#         y_2[:, :, 1::2, 0::2] = 0
#         ctx_params_2 = self.context_prediction_2(y_2)
#         ctx_params_2[:, :, 0::2, 0::2] = 0
#         ctx_params_2[:, :, 1::2, :] = 0
#
#         y_3 = y_hat.clone()
#         y_3[:, :, 1::2, 0::2] = 0
#         ctx_params_3 = self.context_prediction_3(y_3)
#         ctx_params_3[:, :, 0::2, :] = 0
#         ctx_params_3[:, :, 1::2, 1::2] = 0
#
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#
#         x_hat = self.g_s(y_hat)
#         return {
#             "x_hat": x_hat,
#             "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#         }
#
#     def aux_loss(self):
#         """Return the aggregated loss over the auxiliary entropy bottleneck
#         module(s).
#         """
#         aux_loss = sum(
#             m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
#         )
#         return aux_loss
#
#     def parameters(self):
#         """Returns an iterator over the model parameters."""
#         for m in self.children():
#             if isinstance(m, EntropyBottleneck):
#                 continue
#             for p in m.parameters():
#                 yield p
#
#     def aux_parameters(self):
#         """
#         Returns an iterator over the entropy bottleneck(s) parameters for
#         the auxiliary loss.
#         """
#         for m in self.children():
#             if not isinstance(m, EntropyBottleneck):
#                 continue
#             for p in m.parameters():
#                 yield p
#
#     def update(self, scale_table=None, force=False):
#         """Updates the entropy bottleneck(s) CDF values.
#
#         Needs to be called once after training to be able to later perform the
#         evaluation with an actual entropy coder.
#
#         Args:
#             scale_table (bool): (default: None)
#                 check if we need to update the gaussian conditional parameters,
#                 the offsets are only computed and stored when the conditonal model is updated.
#             force (bool): overwrite previous values (default: False)
#
#         Returns:
#             updated (bool): True if one of the EntropyBottlenecks was updated.
#         """
#         if scale_table is None:
#             scale_table = get_scale_table()
#         self.gaussian_conditional.update_scale_table(scale_table, force=force)
#
#         updated = False
#         for m in self.children():
#             if not isinstance(m, EntropyBottleneck):
#                 continue
#             rv = m.update(force=force)
#             updated |= rv
#         return updated
#
#     def load_state_dict(self, state_dict, strict=True):
#         # Dynamically update the entropy bottleneck buffers related to the CDFs
#         update_registered_buffers(
#             self.entropy_bottleneck,
#             "entropy_bottleneck",
#             ["_quantized_cdf", "_offset", "_cdf_length"],
#             state_dict,
#         )
#         update_registered_buffers(
#             self.gaussian_conditional,
#             "gaussian_conditional",
#             ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
#             state_dict,
#         )
#         super().load_state_dict(state_dict, strict=strict)
#
#     def compress(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         y = self.g_a(x)
#         z = self.h_a(y)
#
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#
#         params = self.h_s(z_hat)
#
#         zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
#         )
#         _, means_hat = gaussian_params.chunk(2, 1)
#         y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
#
#         y_1 = y_hat.clone()
#         y_1[:, :, 0::2, 1::2] = 0
#         y_1[:, :, 1::2, :] = 0
#         ctx_params_1 = self.context_prediction_1(y_1)
#         ctx_params_1[:, :, 0::2, :] = 0
#         ctx_params_1[:, :, 1::2, 0::2] = 0
#
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
#         )
#         _, means_hat = gaussian_params.chunk(2, 1)
#         y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
#
#         y_2 = y_hat.clone()
#         y_2[:, :, 0::2, 1::2] = 0
#         y_2[:, :, 1::2, 0::2] = 0
#         ctx_params_2 = self.context_prediction_2(y_2)
#         ctx_params_2[:, :, 0::2, 0::2] = 0
#         ctx_params_2[:, :, 1::2, :] = 0
#
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
#         )
#         _, means_hat = gaussian_params.chunk(2, 1)
#         y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
#
#         y_3 = y_hat.clone()
#         y_3[:, :, 1::2, 0::2] = 0
#         ctx_params_3 = self.context_prediction_3(y_3)
#         ctx_params_3[:, :, 0::2, :] = 0
#         ctx_params_3[:, :, 1::2, 1::2] = 0
#
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#
#         y1, y2, y3, y4 = Demultiplexer(y)
#         scales_hat_y1, scales_hat_y2, scales_hat_y3, scales_hat_y4 = Demultiplexer(scales_hat)
#         means_hat_y1, means_hat_y2, means_hat_y3, means_hat_y4 = Demultiplexer(means_hat)
#
#         indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
#         indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
#         indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
#         indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)
#
#         y1_strings = self.gaussian_conditional.compress(y1, indexes_y1, means=means_hat_y1)
#         y2_strings = self.gaussian_conditional.compress(y2, indexes_y2, means=means_hat_y2)
#         y3_strings = self.gaussian_conditional.compress(y3, indexes_y3, means=means_hat_y3)
#         y4_strings = self.gaussian_conditional.compress(y4, indexes_y4, means=means_hat_y4)
#
#         return {
#             "strings": [y1_strings, y2_strings, y3_strings, y4_strings, z_strings],
#             "shape": z.size()[-2:],
#         }
#
#     def decompress(self, strings, shape):
#         """
#         See Figure 5. Illustration of the proposed two-pass decoding.
#         """
#         assert isinstance(strings, list) and len(strings) == 5
#         z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
#         params = self.h_s(z_hat)
#
#         # Stage 0:
#         zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         scales_hat_y1, _, _, _ = Demultiplexer(scales_hat)
#         means_hat_y1, _, _, _ = Demultiplexer(means_hat)
#
#         indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
#         _y1 = self.gaussian_conditional.decompress(strings[0], indexes_y1, means=means_hat_y1)  # [1, 384, 8, 8]
#         y1 = Multiplexer(_y1, torch.zeros_like(_y1), torch.zeros_like(_y1), torch.zeros_like(_y1))  # [1, 192, 16, 16]
#
#         # Stage 1:
#         ctx_params_1 = self.context_prediction_1(y1)
#         ctx_params_1[:, :, 0::2, :] = 0
#         ctx_params_1[:, :, 1::2, 0::2] = 0
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, scales_hat_y2, _, _ = Demultiplexer(scales_hat)
#         _, means_hat_y2, _, _ = Demultiplexer(means_hat)
#
#         indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
#         _y2 = self.gaussian_conditional.decompress(strings[1], indexes_y2, means=means_hat_y2)  # [1, 384, 8, 8]
#         y2 = Multiplexer(torch.zeros_like(_y2), _y2, torch.zeros_like(_y2), torch.zeros_like(_y2))  # [1, 192, 16, 16]
#
#         # Stage 2:
#         ctx_params_2 = self.context_prediction_2(y1 + y2)
#         ctx_params_2[:, :, 0::2, 0::2] = 0
#         ctx_params_2[:, :, 1::2, :] = 0
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, _, scales_hat_y3, _ = Demultiplexer(scales_hat)
#         _, _, means_hat_y3, _ = Demultiplexer(means_hat)
#
#         indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
#         _y3 = self.gaussian_conditional.decompress(strings[2], indexes_y3, means=means_hat_y3)  # [1, 384, 8, 8]
#         y3 = Multiplexer(torch.zeros_like(_y3), torch.zeros_like(_y3), _y3, torch.zeros_like(_y3))  # [1, 192, 16, 16]
#
#         # Stage 3:
#         ctx_params_3 = self.context_prediction_3(y1 + y2 + y3)
#         ctx_params_3[:, :, 0::2, :] = 0
#         ctx_params_3[:, :, 1::2, 1::2] = 0
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
#         )
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, _, _, scales_hat_y4 = Demultiplexer(scales_hat)
#         _, _, _, means_hat_y4 = Demultiplexer(means_hat)
#
#         indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)
#         _y4 = self.gaussian_conditional.decompress(strings[3], indexes_y4, means=means_hat_y4)  # [1, 384, 8, 8]
#         y4 = Multiplexer(torch.zeros_like(_y4), torch.zeros_like(_y4), torch.zeros_like(_y4), _y4)  # [1, 192, 16, 16]
#
#         # gather
#         y_hat = y1 + y2 + y3 + y4
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#
#         return {
#             "x_hat": x_hat,
#         }



####################################################scctx##############################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d
from timm.models.layers import trunc_normal_
from .layers import ConvNeXtLayer, CheckerboardMaskedConv2d
from .utils import update_registered_buffers, Demultiplexer, Multiplexer, conv, deconv, quantize_ste

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
        min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class cnexTIC(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_chans = config['in_chans']
        embed_dim = config['embed_dim']
        latent_dim = config['latent_dim']
        drop_path_rate = 0.1
        self.num_slices = 5
        N = embed_dim
        M = latent_dim
        self.M = M

        depths = [2, 4, 6, 2, 2, 2]
        # depths = [3, 3, 9, 3, 3, 3]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.g_a0 = ConvNeXtLayer(dim_in=in_chans,
                                  dim_out=embed_dim,
                                  depth=depths[0],
                                  drop = dpr[sum(depths[:0]):sum(depths[:1])],
                                  is_first = True,
                                  encode=True)
        self.g_a1 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[1],
                                  drop = dpr[sum(depths[:1]):sum(depths[:2])],
                                  is_first = False,
                                  encode=True)
        self.g_a2 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[2],
                                  drop = dpr[sum(depths[:2]):sum(depths[:3])],
                                  is_first = False,
                                  encode=True)
        self.g_a3 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=latent_dim,
                                  depth=depths[3],
                                  drop = dpr[sum(depths[:3]):sum(depths[:4])],
                                  is_first = False,
                                  encode=True)
        self.h_a0 = ConvNeXtLayer(dim_in=latent_dim,
                                  dim_out=embed_dim,
                                  depth=depths[4],
                                  drop = dpr[sum(depths[:4]):sum(depths[:5])],
                                  is_first = False,
                                  encode=True,
                                  is_hyper=True)
        self.h_a1 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[5],
                                  drop = dpr[sum(depths[:5]):sum(depths[:6])],
                                  is_first = False,
                                  encode=True,
                                  is_hyper=True)

        depths = depths[::-1]
        self.h_s0 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[0],
                                  drop = dpr[sum(depths[:0]):sum(depths[:1])],
                                  is_first = False,
                                  encode=False,
                                  is_hyper=True)
        self.h_s1 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=2*latent_dim,
                                  depth=depths[1],
                                  drop = dpr[sum(depths[:1]):sum(depths[:2])],
                                  is_first = False,
                                  encode=False,
                                  is_hyper=True)

        self.g_s0 = ConvNeXtLayer(dim_in=latent_dim,
                                  dim_out=embed_dim,
                                  depth=depths[2],
                                  drop = dpr[sum(depths[:2]):sum(depths[:3])],
                                  is_first = False,
                                  encode=False)
        self.g_s1 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[3],
                                  drop = dpr[sum(depths[:3]):sum(depths[:4])],
                                  is_first = False,
                                  encode=False)
        self.g_s2 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=embed_dim,
                                  depth=depths[4],
                                  drop = dpr[sum(depths[:4]):sum(depths[:5])],
                                  is_first = False,
                                  encode=False)
        self.g_s3 = ConvNeXtLayer(dim_in=embed_dim,
                                  dim_out=in_chans,
                                  depth=depths[5],
                                  drop = dpr[sum(depths[:5]):sum(depths[:6])],
                                  is_first = True,
                                  encode=False)

        self.entropy_bottleneck = EntropyBottleneck(embed_dim)
        self.gaussian_conditional = GaussianConditional(None)

        in_ch_list = [0, 16, 32, 64, 128]
        out_ch_list = [16, 16, 32, 64, M - 128]
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(2 * M + in_ch_list[i], 224, stride=1, kernel_size=5),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=5),
                nn.GELU(),
                conv(128, 2 * out_ch_list[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.sc_transforms = nn.ModuleList(
            CheckerboardMaskedConv2d(
                out_ch_list[i], 2 * out_ch_list[i], kernel_size=5, padding=2, stride=1
            ) for i in range(self.num_slices)
        )
        self.entropy_parameters = nn.ModuleList(
            nn.Sequential(
                conv(2 * M + 12 // 3 * out_ch_list[i], 10 // 3 * out_ch_list[i], 1, 1),
                nn.GELU(),
                conv(10 // 3 * out_ch_list[i], 8 // 3 * out_ch_list[i], 1, 1),
                nn.GELU(),
                conv(8 // 3 * out_ch_list[i], 6 // 3 * out_ch_list[i], 1, 1),
            ) for i in range(self.num_slices)
        )
        self.apply(self._init_weights)

        # g_a, encoder

    def g_a(self, x):
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x)
        x = self.g_a3(x)
        return x

    def g_s(self, x):
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s2(x)
        x = self.g_s3(x)
        return x

    def h_a(self, x):
        x = self.h_a0(x)
        x = self.h_a1(x)
        return x

    def h_s(self, x):
        x = self.h_s0(x)
        x = self.h_s1(x)
        return x

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

        y_slices = y.split([16, 16, 32, 64, self.M - 128], 1)
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

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)
                check if we need to update the gaussian conditional parameters,
                the offsets are only computed and stored when the conditonal model is updated.
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

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        y_slices = y.split([16, 16, 32, 64, self.M - 128], 1)
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
            y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols",
                                                                      means_hat_non_anchor)
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
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
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
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2] * 2, z_hat.shape[3] * 2)
            y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

            y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
