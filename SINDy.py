from LorenzAnimation import AnimatedLorenz
from equations.LorenzEquation import lorenz_equation_p
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from signal_processing import ProcessedSignal


def plot_comparison():
    # Plot analytic and spectral derivatives
    t = sig.t
    dxdt_exact = sig.dxdt_exact.values
    dxdt_spectral = sig.dxdt_spectral.values
    dxdt_spectral_filtered = sig.dxdt_spectral_filtered.values
    dxdt_findiff = sig.dxdt_finitediff.values
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
        plt.xlabel('Time t [s]')
        ylabels = [r'$\.{X}_{' + str(i + 1) + '} [s^{-1}]$' for i in range(sig.dims)]
        for ii, ax in enumerate(axs):
            ax.plot(t, dxdt_exact[:, ii], 'k', alpha=1, linewidth=2, label='Exact')
            ax.plot(t, dxdt_spectral[:, ii], '-', color='blue', alpha=0.8, linewidth=2, label='Spectral Cutoff')
            ax.plot(t, dxdt_findiff[:, ii], '-', color='c', alpha=0.5, label='Forward Finite Difference')
            ax.plot(t, dxdt_spectral_filtered[:, ii], '-', color='red', alpha=0.8, linewidth=2,
                    label='Spectral Cutoff Filtered')
            ax.set_ylabel(ylabels[ii])
            ax.legend(loc=1)

tmax = 300
x0 = [15, -17, 12]
sys = AnimatedLorenz(x0, tmax, anim_speed=60, dt=0.0025, noise_strength=0.5)

lorenz = lorenz_equation_p()
lorenz_ti = lambda x: lorenz(0, x)
sig = ProcessedSignal(
    sys.sim_data,
    noise_power=0.5,
    spectral_cutoff=[7500, 7500, 7500],
    kernel='hann',
    kernel_size=32,
    model=lorenz_ti)

plot_comparison()




