from LorenzAnimation import AnimatedLorenz
from equations.LorenzEquation import lorenz_equation_p
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from itertools import combinations
from signal_processing import ProcessedSignal

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

sig.plot_dxdt_comparison()
sig.plot_svd()

# SINDy
dx = sig.dxdt_exact
x = sig.x_clean


