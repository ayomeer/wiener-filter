import numpy as np
# from scipy.signal import convolve2d
from convolution2D import convolve2d
import matplotlib.pyplot as plt

# --- Set up Test image ------------------------------------
M = N = 100
testImg = np.zeros((M, N))

# Add white square in the center
a = 10 # square side length 
whiteIdxRange_m = np.arange((M-a)/2, (M+a)/2).astype(int)
whiteIdxRange_n = np.arange((N-a)/2, (N+a)/2).astype(int)

for m in whiteIdxRange_m:
    for n in whiteIdxRange_n:
        testImg[m, n] = 1

# --- Set up point spread function (PSF) -------------------

# Box Kernel
kernelSize = 5
h_box = (1/kernelSize**2)*np.ones((kernelSize, kernelSize))
H_box = np.fft.fft2(h_box)
outImg = convolve2d(testImg, h_box)
plt.imshow(outImg, cmap='gray')
plt.show()
