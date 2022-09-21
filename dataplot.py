import matplotlib.pyplot as plt
import numpy as np

TIC_bpp = np.array([0.108, 0.177, 0.279, 0.420, 0.595, 0.835, 1.149, 1.528])
TIC_psnr = np.array([28.58, 30.18, 31.78, 33.50, 35.47, 37.33, 39.24, 41.02])
TIC_msssim = np.array([0.9243, 0.9485, 0.9653, 0.9769, 0.9848, 0.9897, 0.9930, 0.9951])

ConvNext_normdown_bpp = np.array([0.283,1.171])
ConvNext_normdown_psnr = np.array([31.56,39.16])
ConvNext_normdown_msssim = np.array([0.9628,0.9927])


ConvNext_bpp = np.array([0.248, 0.858])
ConvNext_psnr = np.array([30.26, 34.19])
ConvNext_msssim = np.array([0.9578, 0.9880])

plt.figure(1)
plt.plot(TIC_bpp,TIC_psnr,marker = 'o')
plt.plot(ConvNext_bpp, ConvNext_psnr, 's')
plt.plot(ConvNext_normdown_bpp, ConvNext_normdown_psnr, '*')

plt.show()