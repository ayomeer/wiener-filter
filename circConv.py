import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# --- Config ------------------------------------------------------------------------
CIRCULAR = True


# --- Setup -------------------------------------------------------------------------
L = 400 # number of samples of signals to convolve
N = 400 # initial number of samples in fourier domain
P = 2*L-1

# Functions to convolve (initial state)
n = np.arange(L) 
f = np.zeros_like(n).astype(float); f[1:300] = 1.0; # f[-50:-1] = 0.5
h = np.zeros_like(n).astype(float); h[1:200] = 0.5; # h[-50:-1] = 0.5
h_ = np.flip(h)

# circular conv in fourier domain

g = np.real(np.fft.ifft(np.fft.fft(f)*np.fft.fft(h)))

def circConvPlot():
    # Set up convolution plot
    gs = gridspec.GridSpec(12,6)
    fig = plt.figure()
    ax_inputs = fig.add_subplot(gs[0:4,:])
    ax_output = fig.add_subplot(gs[5:9,:])
    ax_xSlider = fig.add_subplot(gs[10,:])
    ax_NSlider = fig.add_subplot(gs[11,:])
   
    # visual config
    ax_inputs.set_xlim(-800, 600)
    ax_output.set_xlim(-20, 820)

    # Plot actual functions
    l_f, = ax_inputs.plot(n, f)
    l_h, = ax_inputs.plot(-n, h, color='C2')
    l_g, = ax_output.plot(n, g, color='C1', zorder=1)
    p_g = ax_output.scatter(0, g[0], color='orange', zorder=3)

    ax_output.set_ylim(ymin=-20, ymax=120)

    if CIRCULAR == True:
        # plot copies of spectrum and time signal
        l_hr, = ax_inputs.plot(-n+n.size, h, color='C2', linestyle='--', alpha=0.65)
        l_hl, = ax_inputs.plot(-n-n.size, h, color='C2', linestyle='--', alpha=0.65)
        
        l_gr, = ax_output.plot(n+N, g, color='C1', linestyle='--')


    sl_x = Slider(ax_xSlider, 'x', 0, P, valinit=0, valstep=1)   
    def sl_x_changed(x):
        _, f = l_f.get_data()
        _, h = l_h.get_data()

        # update inputs plot
        l_h.set_xdata(-n+x)

        if CIRCULAR == True:
            l_hl.set_xdata(-n-n.size+x)
            l_hr.set_xdata(-n+n.size+x)

        # update shading
        h_ = np.roll(np.flip(h), x)
        sl_x_changed.c_fill.remove()
        sl_x_changed.c_fill = ax_inputs.fill_between(n, np.minimum(h_, f), alpha=0.4, color='C1')

        # TODO: update indicator on conv output plot
        p_g.set_offsets([x, np.roll(g, -x)[0]]) # roll for repeated spectra

    sl_x_changed.c_fill = ax_inputs.fill_between(n, np.minimum(h_, f), alpha=0.4, color='C1')
    sl_x.on_changed(sl_x_changed)

    sl_N = Slider(ax_NSlider, 'N', N, P, valinit=f.size, valstep=1)
    def sl_N_changed(N):
        global f, h, g, n

        # Update vertical lines indicating nyquist interval
        sl_N_changed.v_g.remove()
        sl_N_changed.v_g = ax_output.vlines([0, N], 0, 120, 'black') # New Nyquist Interval

        # update vlines indicating FFT convolution summation range
        sl_N_changed.v_in.remove()
        sl_N_changed.v_in = ax_inputs.vlines([0, N], 0, 1.2, 'black')

        # Change padding on f,g
        n = np.arange(N)
        f_pad = np.pad(f, (0, N-f.size))
        h_pad = np.pad(h, (0, N-h.size))

        # Update inputs plot
        l_f.set_data(n, f_pad)
        l_h.set_data(-n, h_pad)
        
        if CIRCULAR == True:
            l_hr.set_data(-n+n.size, h_pad)
            l_hl.set_data(-n-n.size, h_pad)

        # update fill
        sl_x_changed(0)

        # recompute convolution in fourier domain -> update output plot
        g = np.real(np.fft.ifft(np.fft.fft(f_pad)*np.fft.fft(h_pad)))
        l_g.set_data(n, g)

        if CIRCULAR == True:
            l_gr.set_data(n+N, g)

    sl_N_changed.v_g = ax_output.vlines([0, N], 0, 120, 'black') # Nyquist Interval
    sl_N_changed.v_in = ax_inputs.vlines([0, N], 0, 1.2, 'black')
    sl_N.on_changed(sl_N_changed)


    plt.show()

if __name__ == '__main__':
    # Compute convolution in time domain for reference
    g_lin = np.convolve(f,h)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(n, f)
    ax1.plot(-n, h, color='C2')
    ax2.plot(np.arange(g_lin.size), g_lin, color='C1')
    plt.show(block=False)

    circConvPlot()