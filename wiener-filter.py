import numpy as np
from scipy.signal import convolve2d
# from convolution2D import convolve2D
import matplotlib.pyplot as plt

# --- Config ------------------------------------------------
H_TYPE = 'box' # degradation function
NOISE_INTENSITY = 0.01


# --- Setup -------------------------------------------------
# -- Test image 
M = N = 100
testImg = np.zeros((M, N))

# Add white square in the center
a = 10 # square side length 
whiteIdxRange_m = np.arange((M-a)/2, (M+a)/2).astype(int)
whiteIdxRange_n = np.arange((N-a)/2, (N+a)/2).astype(int)

for m in whiteIdxRange_m:
    for n in whiteIdxRange_n:
        testImg[m, n] = 1


# -- point spread functions (kernels) 
# Box Kernel
kernelSize = 5
h_box = (1/kernelSize**2)*np.ones((kernelSize, kernelSize))

# Gaussian Blur
# TODO: Implement Gaussian Blur Kernel


# -- Zero Mean (Gaussian) White Noise
n = np.random.normal(0, NOISE_INTENSITY, (M,N))

# -- Degrade Image 
g = convolve2d(testImg, h_box, mode='same') + n

plt.figure(); plt.title('blurred image')
plt.imshow(g, cmap='gray')
plt.show(block=False)


# --- Image Restoration ---------------------------------------- 
if H_TYPE == 'box': 
    h = h_box
if H_TYPE == 'gaussian':
    pass # TODO

# Transform image and degradation functions into fourier domain

# Determine padding P=L+M to avoid circular convolution problem
P = (M + h.shape[0]-1, N + h.shape[1]-1)
 
G = np.fft.fft2(g, P)
H = np.fft.fft2(h, P)

F_hat = G/H
f_hat = np.fft.ifft2(F_hat, P)

plt.figure(); plt.title('unblurred image')
plt.imshow(abs(f_hat), cmap='gray')
plt.show()

