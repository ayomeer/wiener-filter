import numpy as np
from scipy.signal import convolve2d
# from convolution2D import convolve2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider


# --- Config ------------------------------------------------------------
INPUT_IMG = 'generated' # options: 'generated', 'aerial'

H_TYPE = 'box' # options: 'box', 'gaussian', ('sinc')
FILTER_METHOD = 'inverse' # options: 'inverse', 'wienerK', 'wienerC'  

NOISE_ENABLE = True # options: True, False
NOISE_STD_DEV = 0.01 
WIENER_K = NOISE_STD_DEV * 2

# --- Setup ---------------------------------------------------------
# Generate test image
match INPUT_IMG:
    case 'generated':
        # generate white square test image
        M = N = 100
        testImg = np.zeros((M, N))

        # add white square
        a = 10 
        testImg[M//2-a//2:M//2+a//2, N//2-a//2:N//2+a//2] = 1

        # add white line at edge (testing for boundary errors)
        testImg[M//2:-1, -3:-1] = 1

        f = testImg
    
    case 'aerial':
        f = plt.imread('img/aerial_gs.png')
        M, N = f.shape


# Generate point spread functions (kernels) 
# Box Kernel
kernelSize = 5
h_box = (1/kernelSize**2)*np.ones((kernelSize, kernelSize))

# Gaussian Blur
# TODO: Implement Gaussian Blur Kernel

if H_TYPE == 'box': 
    h = h_box
if H_TYPE == 'gaussian':
    pass # TODO 


# Generate noise degradation
match NOISE_ENABLE:
    case True:
        n = np.random.normal(0, NOISE_STD_DEV, (M+h.shape[0]-1, N+h.shape[1]-1))
    case False:
        n = 0

# Apply degradations to test image
g = convolve2d(f, h, mode='full') + n

# --- Image Restoration Functions ---------------------------------------
def inverseFiltering(g, h): 
    # -- Transform image and degradation functions into fourier domain
    P = (g.shape[0] + h.shape[0]-1, g.shape[1] + h.shape[1]-1) # avoid circular conv
    
    G = np.fft.fft2(g, P) # input zero-padded before transform
    H = np.fft.fft2(h, P) 

    F_hat = G/H
    f_hat = np.fft.ifft2(F_hat)

    return np.real(f_hat)

def wienerKFiltering(g, h, K):
    P = (g.shape[0] + h.shape[0]-1, g.shape[1] + h.shape[1]-1) # avoid circular conv
    
    G = np.fft.fft2(g, P)
    H = np.fft.fft2(h, P)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + K)
   
    F_hat = G*W
    f_hat = np.fft.ifft2(F_hat)
    
    return np.real(f_hat)


NOISE_POWER = (M*N)**2*NOISE_STD_DEV**2 # TODO: get this from local neighborhood variance
def wienerCFiltering(g, h, gamma):
    P = (g.shape[0] + h.shape[0]-1, g.shape[1] + h.shape[1]-1) # avoid circular conv

    G = np.fft.fft2(g, P)
    H = np.fft.fft2(h, P)

    u, v = np.arange(P[1]), np.arange(P[0]) # shape interpreted differntly(!)
    uu, vv = np.meshgrid(u, v)
    C = -4*np.pi**2*(uu**2+vv**2)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + gamma*np.abs(C)**2)

    F_hat = G*W

    phi = sum(sum(abs(G-H*F_hat)**2))
    f_hat = np.fft.ifft2(F_hat) 
    
    return np.real(f_hat), phi

# --- Helper Functions --------------------------------------------------
def getMinNeighborhoodVariance(g):
    pass # TODO implement local variances


# --- MAIN -------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Show original- and distorted image
    fig0 = plt.figure(figsize=(7,9))
    im0 = fig0.add_subplot(211); im0.set_title('original image')
    im1 = fig0.add_subplot(212); im1.set_title('degraded (blurred + noise)')
    im0.imshow(f, cmap='gray')
    im1.imshow(g, cmap='gray')
    plt.show(block=False)

    # --- Image Restoration -------------------------------------------
    match FILTER_METHOD:
        case 'inverse':
            f_hat = inverseFiltering(g, h)

            plt.figure(); plt.title('unblurred image')
            plt.imshow(f_hat[0:M, 0:N], cmap='gray')
            plt.show()
    
        case 'wienerK':
            # Set up figure
            gs = gridspec.GridSpec(10,4)
            fig = plt.figure(figsize=(6,7))
            axImg = fig.add_subplot(gs[0:8,:])
            
            # Set up slider to change K interactively
            axSlider = fig.add_subplot(gs[9,:])
            sl = Slider(axSlider, 'K', 0, 0.2, valinit=0, valstep=0.001)

            def sl_changed(sliderK):
                f_hat = wienerKFiltering(g, h, sliderK)
                im.set_data(f_hat[0:M, 0:N])

            sl.on_changed(sl_changed)

            # Do filtering and show result
            f_hat = wienerKFiltering(g, h, WIENER_K)
            im = axImg.imshow(f_hat[0:M, 0:N], cmap='gray')
            
            
        case 'wienerC':
            # Set up figure
            gs = gridspec.GridSpec(10,4)
            fig = plt.figure(figsize=(6,7))
            axImg = fig.add_subplot(gs[0:8,:])
            fig.text(0.6, 0.95, "noise power = {:e}".format(NOISE_POWER))
            t = fig.text(0.1, 0.95 ," ")

            # Set up slider to change gamma interactively
            axSlider = fig.add_subplot(gs[9,:])

            if INPUT_IMG == 'generated': gamma_mag = -12 
            elif INPUT_IMG =='aerial': gamma_mag = -16

            sl = Slider(axSlider, 'gamma', 0, 10**(gamma_mag), valinit=0, valstep=10**((gamma_mag-3)))

            def sl_changed(sliderVal):
                f_hat, phi = wienerCFiltering(g, h, sliderVal)
                im.set_data(f_hat[0:M, 0:N])
                t.set_text(r"$\phi(\gamma)$ = {:e}".format(phi))

            sl.on_changed(sl_changed)

            # Do filtering and show result
            f_hat, _ = wienerCFiltering(g, h, gamma=0) # initial plot: regular inverse filtering
            im = axImg.imshow(np.clip(f_hat, 0, 1)[0:M, 0:N], cmap='gray')

    plt.show(block=True)



