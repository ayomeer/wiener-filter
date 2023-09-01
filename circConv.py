import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# --- Config ------------------------------------------------------------------------



# --- Setup -------------------------------------------------------------------------

N = 400
n = np.arange(N)

# Functions to convolve
f = np.zeros_like(n).astype(float); f[1:300] = 1.0
h = np.zeros_like(n).astype(float); h[1:200] = 0.5

g = np.fft.ifft(np.fft.fft(f)*np.fft.fft(h)) # circular conv in fourier domain

def plot():
        # Set up convolution visualization w/ slider
    gs = gridspec.GridSpec(12,6)
    fig = plt.figure()
    ax_inputs = fig.add_subplot(gs[0:4,:])
    ax_output = fig.add_subplot(gs[5:9,:])
    ax_xSlider = fig.add_subplot(gs[10,:])
    ax_NSlider = fig.add_subplot(gs[11,:])
   
    ax_inputs.vlines([0, 400], 0, 1.2, 'black')
    ax_inputs.plot(n, f)
    l_h,  = ax_inputs.plot(-n, h, 'b-')
    # l_hl, = ax_inputs.plot(-n-n.size, h, 'b--')
    l_hr, = ax_inputs.plot(-n+n.size, h, 'b--')

    h_ = np.flip(h)
    # c_fill = ax_inputs.fill_between(n, np.minimum(h_, f))

    l_g, = ax_output.plot(n, g, color='C1'); ax_output.set_ylim(ymin=0)
    

    sl_x = Slider(ax_xSlider, 'x', 0, N, valinit=0, valstep=1)
    sl_N = Slider(ax_NSlider, 'N', 400, n.size*2, valinit=400, valstep=1)

    
    def sl_x_changed(x):
        # update inputs plot
        l_h.set_xdata(-n+x)
        # l_hl.set_xdata(-n-n.size+x)
        l_hr.set_xdata(-n+n.size+x)

        # update shading
        h_ = np.roll(np.flip(h), x)
        sl_x_changed.c_fill.remove()
        sl_x_changed.c_fill = ax_inputs.fill_between(n, np.minimum(h_, f), alpha=0.4, color='C1')

        # TODO: update indicator on conv output plot

        fig.canvas.draw()
        fig.canvas.flush_events()
    sl_x_changed.c_fill = ax_inputs.fill_between(n, np.minimum(h_, f), alpha=0.4, color='C1')
    
    def sl_N_changed(N):
        # TODO: change padding on f,g -> update inputs plot
        # TODO: recompute convolution in fourier domain -> update output plot
        pass

    sl_x.on_changed(sl_x_changed)
    sl_N.on_changed(sl_N_changed)

    plt.show()

if __name__ == '__main__':
    plot()