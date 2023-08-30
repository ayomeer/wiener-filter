import numpy as np
import matplotlib.pyplot as plt

Lf = 5
Lh = 3; M = 2

f = np.array([0, 0, 1, 0, 0])
h = (1/3)*np.array([1, 1, 1])

# -- convolution in space domain (linear)
y = np.convolve(f, h)
Ly = y.size

plt.plot(np.arange(Ly), y)
plt.show(block=False)


# -- convolution in freq domain (circular)
F = np.fft.fft(f)
H = np.fft.fft(h)
H = np.append(H, [0, 0])

Y = F*H
y = np.fft.ifft(Y)
Ly = y.size

plt.plot(np.arange(Ly), y)
plt.show(block=False)


# -- convolution in freq domain w/ padding before FFT (linear)
f = np.append(f, np.zeros(M)) 
h = np.append(h, np.zeros(Lf-1))

F = np.fft.fft(f)
H = np.fft.fft(h)

Y = F*H
y = np.fft.ifft(Y)
Ly = y.size

plt.plot(np.arange(Ly), y)
plt.show()
