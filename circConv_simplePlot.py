import numpy as np
import matplotlib.pyplot as plt

# --- Config ------------------------------------------------------------------------



# --- Setup -------------------------------------------------------------------------
Lf = 5
Lh = 3; M = 2

# Functions to convolve
f = np.array([0, 0, 1, 0, 0])
h = (1/3)*np.array([1, 1, 1])



if __name__ == '__main__':

    # -- convolution in space domain (linear)
    y_lin = np.convolve(f, h)

    plt.plot(np.arange(y_lin.size), y_lin)

    # -- convolution in freq domain (circular)
    F = np.fft.fft(f)
    H = np.fft.fft(h)
    H = np.append(H, [0, 0])

    Y = F*H
    y_circ = np.fft.ifft(Y)

    plt.plot(np.arange(y_circ.size), y_circ)

    # -- convolution in freq domain w/ padding before FFT (linear)
    f = np.append(f, np.zeros(M)) 
    h = np.append(h, np.zeros(Lf-1))

    F = np.fft.fft(f)
    H = np.fft.fft(h)

    Y = F*H
    y_circ_padded = np.fft.ifft(Y)

    plt.plot(np.arange(y_circ_padded.size), y_circ_padded)
    plt.show()
