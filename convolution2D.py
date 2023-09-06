import numpy as np
import matplotlib.pyplot as plt

# --- Constants ----------------------------------------------------------
PI = np.pi

# --- Config -------------------------------------------------------------
KERNEL = 'gaussian' # options: 'box', 'gaussian' 
PLOT_INTERMEDIATE = True

# --- Setup Kernel -------------------------------------------------------

# Gauss Kernel
kernelSize_gauss = 9
r_gauss = kernelSize_gauss//2
x = y = np.linspace(-r_gauss, r_gauss, kernelSize_gauss)
xx, yy = np.meshgrid(x, y)

sigma = 1
w_gauss = (1/(PI*2*sigma**2)) * np.exp(-(xx**2+yy**2)/(2*sigma**2))

w_gauss_x = (1/(np.sqrt(PI*2*sigma**2))) * np.exp(-(x**2)/(2*sigma**2))
w_gauss_y = (1/(np.sqrt(PI*2*sigma**2))) * np.exp(-(y**2)/(2*sigma**2))

# s = sum(sum(w_gauss_x[:, np.newaxis]@w_gauss_y[np.newaxis, :])) # check sum = 1
# --- Config -------------------------------------------------------------

# 2D Convolution as separated 1D Convolutions
def convolve2D(img, w1, w2, plotIntermediate=False):
    M, N = img.shape
    
    # First set of convolution, horizontally, row by row
    g = np.zeros((M,N))
    for m in range(M):  
        g[m] = np.convolve(img[m,:], w1, mode='same')
    
    if plotIntermediate == True:
        plt.figure(); plt.title('seperable conv intermediate result')
        plt.imshow(g, cmap='gray')
        plt.show(block=False)

    # Second set of convolutions, vertically, column by column
    g2 = np.zeros((M,N))
    for n in range(N):
        g2[:,n] = np.convolve(g[:,n], w2, mode='same')

    return g2


if __name__ == '__main__':
    # test example
    from scipy.signal import convolve2d

    img = np.zeros((10,10))
    img[4:6, 4:6] = 1
    plt.imshow(img, cmap='gray')
    plt.show(block=False)

    match KERNEL:
        case 'box':
            w = np.ones((3,3))*1/9 # 3x3 Box Kernel
            w1 = np.ones((1,3)).squeeze()*1/3 # "column vector"
            w2 = np.ones((1,3)).squeeze()*1/3

        case 'gaussian':
            w = w_gauss
            w1 = w_gauss_x
            w2 = w_gauss_y

    g = convolve2d(img, w, mode='same')
    g_sep = convolve2D(img, w1, w2, plotIntermediate=PLOT_INTERMEDIATE)  

    plt.figure(); plt.title('convoltion output 2D')
    plt.imshow(g, cmap='gray')
    plt.show(block=False)

    plt.figure(); plt.title('convoltion output 1D*1D')
    plt.imshow(g_sep, cmap='gray')
    plt.show()