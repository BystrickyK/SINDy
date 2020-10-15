import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from LorenzAnimation import AnimatedLorenz
from dynamical_systems import LorenzSystem, DynamicalSystem, LotkaVolterraSystem
from equations.Lorenz import lorenz_equation
from equations.LotkaVolterra import lotka_volterra_equation
from function_library import poly_library
from signal_processing import StateSignal, ForcingSignal, ProcessedSignal
import time as time
from utils.regression import seq_thresh_ls
from utils.visualization import plot_ksi, plot_svd, plot_dxdt_comparison
from utils.tools import cutoff

# Simulate the dynamical system
# tmax = 120
# x0 = [-15, 30, -5]
# sys = LorenzSystem(x0, dt=0.0025)
# sys.propagate(tmax)
sys = LotkaVolterraSystem([2, 4])
sys.propagate(400)
# Forcing data
u = ForcingSignal(sys.sim_data[:, [0, 3, 4]])
# Load the lorenz system function for analytical derivative computation
# model = lorenz_equation_p()
model = lotka_volterra_equation()
# System dimensionality
dims = (sys.sim_data.shape[1]-1)//2
# Create a ProcessedSignal instance - calculate derivatives, filter out noise etc.
sig = ProcessedSignal(
    sys.sim_data[:, 0:dims+1],
    noise_power=0,
    spectral_cutoff=0.1,
    kernel='flattop',
    kernel_size=64,
    model=model
)

# Plot derivatives comparison and SVD
plot_dxdt_comparison(sig)
plot_svd(sig.svd)

# %%
# SINDy
fig, ax = plt.subplots(1, 1, tight_layout=True)
dx = sig.dxdt_spectral_filtered
x = sig.x_filtered
# dx = sig.dxdt_exact
# x = sig.x_clean
theta = poly_library(x, poly_orders=(1, 2, 3, 4))

dx = cutoff(dx, sig.kernel_size)
x = cutoff(x, sig.kernel_size)
theta = cutoff(theta, sig.kernel_size)

ksi = seq_thresh_ls(theta, dx, n=50, alpha=0.1, verbose=True, threshold=0.5)
plot_ksi(ksi, theta, dx, ax, show_sparse=True)
