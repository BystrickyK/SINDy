import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from LorenzAnimation import AnimatedLorenz
from dynamical_systems import LorenzSystem
from equations.LorenzEquation import lorenz_equation_p
from function_library import poly_library
from signal_processing import ProcessedSignal
import time as time
from utils.regression import seq_thresh_ls
from utils.visualization import plot_ksi, plot_svd, plot_dxdt_comparison
from utils.tools import cutoff

# Simulate the dynamical system
tmax = 120
x0 = [-15, 30, -5]
sys = LorenzSystem(x0, dt=0.0025)
sys.propagate(tmax)

# Load the lorenz system function for analytical derivative computation
lorenz = lorenz_equation_p()
# Create a ProcessedSignal instance - calculate derivatives, filter out noise etc.
sig = ProcessedSignal(
    sys.sim_data,
    noise_power=0.25,
    spectral_cutoff=[0.3, 0.3, 0.3],
    kernel='flattop',
    kernel_size=64,
    model=lambda x: lorenz(0, x))

# Plot derivatives comparison and SVD
plot_dxdt_comparison(sig)
plot_svd(sig.svd)

# %%
# SINDy
fig, ax = plt.subplots(1, 1, tight_layout=True)
dx = sig.dxdt_spectral_filtered
x = sig.x_filtered
dx = sig.dxdt_exact
x = sig.x_clean
theta = poly_library(x, poly_orders=(1, 2, 3, 4, 5))

dx = cutoff(dx, sig.kernel_size)
x = cutoff(x, sig.kernel_size)
theta = cutoff(theta, sig.kernel_size)

ksi_ridge = seq_thresh_ls(theta, dx, n=50, alpha=0, verbose=True, threshold=0.005)
plot_ksi(ksi_ridge, theta, dx, ax)
