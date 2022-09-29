import matplotlib.pyplot as plt
import numpy as np

TIC_bpp = np.array([0.108, 0.177, 0.279, 0.420, 0.595, 0.835, 1.149, 1.528])
TIC_psnr = np.array([28.58, 30.18, 31.78, 33.50, 35.47, 37.33, 39.24, 41.02])
TIC_msssim = np.array([0.9243, 0.9485, 0.9653, 0.9769, 0.9848, 0.9897, 0.9930, 0.9951])

ConvNext_normdown_bpp = np.array([0.283,1.171])
ConvNext_normdown_psnr = np.array([31.56,39.16])
ConvNext_normdown_msssim = np.array([0.9628,0.9927])

ConvNext_scctx_bpp = np.array([0.319,1.218])
ConvNext_scctx_psnr = np.array([32.34,39.57])
ConvNext_scctx_msssim = np.array([0.9687,0.9934])

ConvNext_AdamW_bpp = np.array([0.318,1.228])
ConvNext_AdamW_psnr = np.array([32.28,39.44])
ConvNext_AdamW_msssim = np.array([0.9681,0.9932])

plt.figure(1)
plt.plot(TIC_bpp,TIC_psnr,marker = 'o')
plt.plot(ConvNext_scctx_bpp, ConvNext_scctx_psnr, 's')
plt.plot(ConvNext_normdown_bpp, ConvNext_normdown_psnr, 's')
plt.plot(ConvNext_AdamW_bpp, ConvNext_AdamW_psnr, 's')
plt.show()