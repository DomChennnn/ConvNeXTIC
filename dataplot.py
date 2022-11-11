import matplotlib.pyplot as plt
import numpy as np

# ############################################# TIC AND ConvNeXtIC######################################################
TIC_bpp = np.array([0.108, 0.177, 0.279, 0.420, 0.595, 0.835, 1.149, 1.528])
TIC_psnr = np.array([28.58, 30.18, 31.78, 33.50, 35.47, 37.33, 39.24, 41.02])
TIC_msssim = np.array([0.9243, 0.9485, 0.9653, 0.9769, 0.9848, 0.9897, 0.9930, 0.9951])

ConvNext_normdown_bpp = np.array([0.283, 1.171])
ConvNext_normdown_psnr = np.array([31.56, 39.16])
ConvNext_normdown_msssim = np.array([0.9628, 0.9927])

ConvNext_scctx_bpp = np.array([0.319, 1.218])
ConvNext_scctx_psnr = np.array([32.34, 39.57])
ConvNext_scctx_msssim = np.array([0.9687, 0.9934])

# ConvNext_AdamW_bpp = np.array([0.318, 1.228])
# ConvNext_AdamW_psnr = np.array([32.28, 39.44])
# ConvNext_AdamW_msssim = np.array([0.9681, 0.9932])

ConvNext_lc_bpp = np.array([0.128,0.205,0.318,0.467,0.660,0.909,1.224,1.694])
ConvNext_lc_psnr = np.array([29.16,30.72,32.51,34.23,36.02,37.80,39.59,40.68])
ConvNext_lc_msssim = np.array([0.9318,0.9532,0.9692,0.9794,0.9860,0.9905,0.9935,0.9950])

# ############################################# cheng2020 and cheng2020-attn######################################################
#
# psnr_attn = np.array(
#     [
#         28.43102333976717,
#         29.765925971604506,
#         31.321335416187427,
#         33.36386083004006,
#         34.9494309886167,
#         36.621812638892735,
#     ]
# )
# bpp_attn = np.array(
#     [
#         0.11557685004340279,
#         0.17447916666666666,
#         0.26903279622395837,
#         0.4269612630208333,
#         0.5952894422743056,
#         0.8058878580729166,
#     ]
# )
#
# psnr_cheng2020 = np.array([
#       28.58235164600031,
#       29.969272906591517,
#       31.342783743928436,
#       33.390308674052186,
#       35.11886905407247,
#       36.7040956894179
#     ])
# bpp_cheng2020 = np.array([
#       0.11959499782986115,
#       0.18379720052083337,
#       0.2709520128038195,
#       0.4173753526475694,
#       0.5944688585069444,
#       0.805735270182292
#     ])




plt.figure(1)
plt.plot(TIC_bpp, TIC_psnr, marker="s", label = 'TIC(400 epoch)')
# plt.plot(bpp_cheng2020,psnr_cheng2020,marker = '*')
# plt.plot(bpp_attn, psnr_attn, marker='o')
plt.plot(ConvNext_normdown_bpp, ConvNext_normdown_psnr, "s", label = 'Convnext(400 epoch)')
plt.plot(ConvNext_scctx_bpp, ConvNext_scctx_psnr, "*", label = 'Convnext+scctx(400 epoch)')
# plt.plot(ConvNext_AdamW_bpp, ConvNext_AdamW_psnr, "s")
plt.plot(ConvNext_lc_bpp, ConvNext_lc_psnr, marker = "s", label = 'Convnext+scctx+larger channel(100 epoch)')
plt.xlabel('BPP')
plt.ylabel('PSNR')
plt.title('Kodak')
plt.legend()
plt.show()

