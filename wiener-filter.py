import numpy as np
from scipy.signal import convolve2d
# from convolution2D import convolve2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# --- Constants ---------------------------------------------------------
PI = np.pi

# --- Config ------------------------------------------------------------
# Input
INPUT_IMG = 'aerial' # options: 'generated', 'aerial'
GEN_TYPE = 'square' # options: 'half-half', 'square'

# Blurring Degradation
CONV_MODE = 'valid' # options: 'valid', 'same', 'full'

H_TYPE = 'gaussian' # options: 'box'(ringing!->artifacts), 'gaussian'
STD_DEV_GAUSS_KERNEL = 1
VIZ_KERNEL = False

# Noise Degradation
NOISE_ENABLE = True # options: True, False
NOISE_STD_DEV = 0.01 
WIENER_K = 0

# Restoration Filtering
PAD_METHOD = 'mirror' # options: 'none', 'zero', 'mirror' (none=>circConv)
CROP_RESULT = False # options: True, False (valid result)
FILTER_METHOD = 'wienerC' # options: 'inverse', 'wienerK', 'wienerC'  

WIENER_K_SLIDERMAX = 0.1
WIENER_K_SLIDERSTEP = 0.001


# --- Setup ---------------------------------------------------------
# Generate test image
match INPUT_IMG:
    case 'generated':
        # generate white square test image
        M = N = 100
        testImg = np.zeros((M, N))

        match GEN_TYPE:
            case 'square':
                # add white square
                a = 6
                testImg[M//2-a//2:M//2+a//2, N//2-a//2:N//2+a//2] = 1

                # add white line at edge (testing for boundary errors)
                testImg[M//2:-10, -5:] = 1
            
            case 'half-half':
                testImg[M//2:,:] = 1

        f = testImg
    
    case 'aerial':
        f = plt.imread('img/aerial_gs.png')
        M, N = f.shape


# Generate point spread functions (kernels) 
# Box Kernel
kernelSize = 5
h_box = (1/kernelSize**2)*np.ones((kernelSize, kernelSize))

# 5x5 Gaussian Blur Kernel from the web
# (https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm)
h_gauss_int = (1/273) * np.array([[1, 4,  7,  4,  1],
                                  [4, 16, 26, 16, 4],
                                  [7, 26, 41, 26, 7],
                                  [4, 16, 26, 16, 4],
                                  [1, 4,  7,  4,  1]])

# 9x9 Gaussian Blur kernel drawn from distribution
kernelSize_gauss = 9
r_gauss = kernelSize_gauss//2
x = y = np.linspace(-r_gauss, r_gauss, kernelSize_gauss)
xx, yy = np.meshgrid(x, y)

sigma = STD_DEV_GAUSS_KERNEL
h_gauss = (1/(PI*2*sigma**2)) * np.exp(-(xx**2+yy**2)/(2*sigma**2))

# Visualize Kernel
if VIZ_KERNEL == True:
    fig = plt.figure(); plt.title('Convolution Kernel')
    ax3D = fig.add_subplot(projection='3d')
    ax3D.plot_surface(xx, yy, h_gauss)
    plt.show(block=False)


# Apply degradation function to image
if H_TYPE == 'box': 
    h = h_box
if H_TYPE == 'gaussian':
    h = h_gauss

g = convolve2d(f, h, mode=CONV_MODE) 

# Add (white) noise
if NOISE_ENABLE == True:
    g += np.clip(np.random.normal(0, NOISE_STD_DEV, g.shape), 0, 1)

# Padding (prep for inverse filtering in fourier domain)
P = (g.shape[0]+h.shape[0]-1, g.shape[1]+h.shape[1]-1)   
match PAD_METHOD:
    case 'none':
        g_pad = g

    case 'zero':
        g_pad = np.pad(g, ((0, h.shape[0]-1), (0, h.shape[1]-1))) # pad at end
    
    case 'mirror':
        g_pad = np.hstack((g, np.fliplr(g)))
        g_pad = np.vstack((g_pad, np.flipud(g_pad)))



# --- Restoration Filter Implementations ----------------------------------------------
def inverseFiltering(g, h): 
    # -- Transform image and degradation functions into fourier domain
    G = np.fft.fft2(g)
    H = np.fft.fft2(h, g.shape)

    F_hat = G/H
    f_hat = np.fft.ifft2(F_hat)

    return np.clip(np.real(f_hat), 0, 1)

def wienerKFiltering(g, h, K):
    G = np.fft.fft2(g)
    H = np.fft.fft2(h, g.shape)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + K)
   
    F_hat = G*W
    f_hat = np.fft.ifft2(F_hat)
    
    return np.clip(np.real(f_hat), 0, 1)


NOISE_POWER = (M*N)**2*NOISE_STD_DEV**2 
def wienerCFiltering(g, h, gamma):
    G = np.fft.fft2(g)
    H = np.fft.fft2(h, g.shape)

    u, v = np.arange(g.shape[1]), np.arange(g.shape[0]) # shape interpreted differntly(!)
    uu, vv = np.meshgrid(u, v)
    C = -4*np.pi**2*(uu**2+vv**2)

    W = (1/H) * (np.abs(H)**2)/(np.abs(H)**2 + gamma*np.abs(C)**2)

    F_hat = G*W

    phi = sum(sum(abs(G-H*F_hat)**2))
    f_hat = np.fft.ifft2(F_hat) 
    
    return np.clip(np.real(f_hat), 0, 1), phi


# --- MAIN -------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Show original- and distorted image
    fig0 = plt.figure(figsize=(10,5))
    im0 = fig0.add_subplot(131); im0.set_title(r'original image $f(x,y)$')
    im1 = fig0.add_subplot(132); im1.set_title(r'degraded image $g(x,y)$')
    im2 = fig0.add_subplot(133); im2.set_title('degraded + padded')
    im0.imshow(f, cmap='gray')
    im1.imshow(g, cmap='gray')
    im2.imshow(g_pad, cmap='gray')
    plt.show(block=False)

    # --- Image Restoration -------------------------------------------
    if CROP_RESULT == True:
        M, N = g.shape[0]-h.shape[0]-1, g.shape[1]-h.shape[1]-1
    else:
        M, N = g.shape

    match FILTER_METHOD:
        case 'inverse':
            f_hat = inverseFiltering(g_pad, h)

            plt.figure(); plt.title(r'Restoration filter output $\hat{f}(x,y)$')
            plt.imshow(f_hat[0:M, 0:N], cmap='gray')
            plt.show()
    
        case 'wienerK':
            # Set up figure
            gs = gridspec.GridSpec(10,4)
            fig = plt.figure(figsize=(6,7))
            axImg = fig.add_subplot(gs[0:8,:])
            
            # Set up slider to change K interactively
            axSlider = fig.add_subplot(gs[9,:])
            sl = Slider(axSlider, 'K', 0, WIENER_K_SLIDERMAX, valinit=0, valstep=WIENER_K_SLIDERSTEP)

            def sl_changed(sliderK):
                f_hat = wienerKFiltering(g_pad, h, sliderK)
                im.set_data(f_hat[0:M, 0:N])

            sl.on_changed(sl_changed)

            # Do filtering and show result (first run)
            f_hat = wienerKFiltering(g_pad, h, 0)
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
            elif INPUT_IMG =='aerial': gamma_mag = -17

            sl = Slider(axSlider, 'gamma', 0, 10**(gamma_mag), valinit=0, valstep=10**((gamma_mag-3)))

            def sl_changed(sliderVal):
                f_hat, phi = wienerCFiltering(g_pad, h, sliderVal)
                im.set_data(f_hat[0:M, 0:N])
                t.set_text(r"$\phi(\gamma)$ = {:e}".format(phi))

            sl.on_changed(sl_changed)

            # Do filtering and show result
            f_hat, _ = wienerCFiltering(g_pad, h, gamma=0) # initial plot: regular inverse filtering
            im = axImg.imshow(np.clip(f_hat, 0, 1)[0:M, 0:N], cmap='gray')

    plt.show(block=True)



