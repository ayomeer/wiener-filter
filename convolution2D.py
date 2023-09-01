import numpy as np
import matplotlib.pyplot as plt

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

    w = np.ones((3,3))*1/9 # 3x3 Box Kernel
    w1 = np.ones((1,3)).squeeze()*1/3 # "column vector"
    w2 = np.ones((1,3)).squeeze()*1/3

    g = convolve2d(img, w, mode='same')
    g_sep = convolve2D(img, w1, w2, plotIntermediate=False)  

    plt.figure(); plt.title('convoltion output 2D')
    plt.imshow(g, cmap='gray')
    plt.show(block=False)

    plt.figure(); plt.title('convoltion output 1D*1D')
    plt.imshow(g_sep, cmap='gray')
    plt.show()