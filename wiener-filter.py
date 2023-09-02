import numpy as np
from scipy.signal import convolve2d
# from convolution2D import convolve2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# --- Config ------------------------------------------------------------
H_TYPE = 'box' # options: 'box', 'gaussiain', ('sinc')
FILTER_METHOD = 'wienerC' # options: 'inverse', 'wienerK', 'wienerC'  

NOISE_STD_DEV = 0.01
WIENER_K = NOISE_STD_DEV * 2

# --- Setup ---------------------------------------------------------
# Generate test image
M = N = 100
testImg = np.zeros((M, N))

a = 10 # white square side length 
whiteIdxRange_m = np.arange((M-a)/2, (M+a)/2).astype(int)
whiteIdxRange_n = np.arange((N-a)/2, (N+a)/2).astype(int)

for m in whiteIdxRange_m:
    for n in whiteIdxRange_n:
        testImg[m, n] = 1


# Generate point spread functions (kernels) 
# Box Kernel
kernelSize = 5
h_box = (1/kernelSize**2)*np.ones((kernelSize, kernelSize))

# Gaussian Blur
# TODO: Implement Gaussian Blur Kernel


# Generate noise degradation
n = np.random.normal(0, NOISE_STD_DEV, (M,N))

# Apply degradations to test image
g = convolve2d(testImg, h_box, mode='same') + n

# --- Helper Functions --------------------------------------------------
def getMinNeighborhoodVariance(g):
    pass # TODO implement local variances

# --- Image Restoration Functions ---------------------------------------
def inverseFiltering(g, h): 
    # -- Transform image and degradation functions into fourier domain
    P = (M + h.shape[0]-1, N + h.shape[1]-1) # avoid circular conv
    
    G = np.fft.fft2(g, P) # input zero-padded before transform
    H = np.fft.fft2(h, P) 

    F_hat = G/H
    f_hat = np.fft.ifft2(F_hat, P)

    return f_hat

def wienerKFiltering(g, h, K):
    P = (M + h.shape[0]-1, N + h.shape[1]-1) # avoid circular conv
    
    G = np.fft.fft2(g, P)
    H = np.fft.fft2(h, P)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + K)
   
    F_hat = G*W
    f_hat = np.fft.ifft2(F_hat, P)
    
    return np.real(f_hat)


NOISE_POWER = (M*N)**2*NOISE_STD_DEV**2
def wienerCFiltering(g, h, gamma):
    P = (M + h.shape[0]-1, N + h.shape[1]-1) # avoid circular conv

    G = np.fft.fft2(g, P)
    H = np.fft.fft2(h, P)

    u, v = np.arange(P[0]), np.arange(P[1])
    uu, vv = np.meshgrid(u, v)
    C = -4*np.pi**2*(uu**2+vv**2)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + gamma*np.abs(C)**2)

    F_hat = G*W

    phi = sum(sum(abs(G-H*F_hat)**2))

    f_hat = np.fft.ifft2(F_hat, P)
    
    return np.real(f_hat), phi


if __name__ == '__main__':

    # Show degraded image to be restored
    plt.figure(); plt.title('blurred image')
    plt.imshow(g, cmap='gray')
    plt.show(block=False)


    # --- Image Restoration -------------------------------------------
    if H_TYPE == 'box': 
        h = h_box
    if H_TYPE == 'gaussian':
        pass # TODO 
    

    match FILTER_METHOD:
        case 'inverse':
            f_hat = inverseFiltering(g, h)

            plt.figure(); plt.title('unblurred image')
            plt.imshow(abs(f_hat), cmap='gray')
            plt.show()
    
        case 'wienerK':
            # Set up figure
            gs = gridspec.GridSpec(10,4)
            fig = plt.figure()
            axImg = fig.add_subplot(gs[0:8,:])
            
            # Set up slider to change K interactively
            axSlider = fig.add_subplot(gs[9,:])
            sl = Slider(axSlider, 'K', 0, 0.2, valinit=0, valstep=0.001)

            def sl_changed(sliderK):
                f_hat = wienerKFiltering(g, h, sliderK)
                im.set_data(f_hat)

            sl.on_changed(sl_changed)

            # Do filtering and show result
            f_hat = wienerKFiltering(g, h, WIENER_K)
            im = axImg.imshow(f_hat, cmap='gray')
            
        case 'wienerC':
            # Set up figure
            gs = gridspec.GridSpec(10,4)
            fig = plt.figure()
            axImg = fig.add_subplot(gs[0:8,:])
            fig.text(0.6, 0.95, "noise power = {}".format(NOISE_POWER))
            t = fig.text(0.1, 0.95 ," ")

            # Set up slider to change K interactively
            axSlider = fig.add_subplot(gs[9,:])
            sl = Slider(axSlider, 'gamma', 0, 1e-12, valinit=0, valstep=1e-15)

            def sl_changed(sliderVal):
                f_hat, phi = wienerCFiltering(g, h, sliderVal)
                im.set_data(f_hat)
                t.set_text("phi = {}".format(phi))

            sl.on_changed(sl_changed)

            # Do filtering and show result
            f_hat, _ = wienerCFiltering(g, h, 0)
            im = axImg.imshow(np.clip(f_hat, 0, 1), cmap='gray')

    plt.show(block=True)



