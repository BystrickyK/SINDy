import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from LorenzAnimation import AnimatedLorenz
from dynamical_systems import LorenzSystem, DynamicalSystem, LotkaVolterraSystem
from equations.Lorenz import lorenz_equation
from equations.LotkaVolterra import lotka_volterra_equation
from function_library_creators import poly_library
from signal_processing import StateSignal, ForcingSignal, ProcessedSignal
import time as time
from utils.regression import seq_thresh_ls
from utils.visualization import plot_ksi, plot_svd, plot_dxdt_comparison
from utils.tools import cutoff
import pandas as pd

# Simulate the dynamical system
# tmax = 120
# x0 = [-15, 30, -5]
# sys = LorenzSystem(x0, dt=0.0025)
# sys.propagate(tmax)
sys = LorenzSystem([-8, 8, 28], dt=0.001)
u1 = lambda t, x: 1*(26-x[0]) + np.random.randn()*0
u2 = lambda t, x: 1*(5-x[1]) + np.random.randn()*0
u3 = lambda t, x: 30*np.sin(3*t)
u = (u1, u2, u3)
sys.propagate(30)
sys.propagate_forced(30, u)
# sys.propagate_forced(20, u)
# Time data
time_data = sys.sim_data[:, 0]
# State data
state_data = StateSignal(time_data, sys.sim_data[:, [1, 2, 3]])
# Forcing data
forcing_data = ForcingSignal(time_data, sys.sim_data[:, [4, 5, 6]])
state_data.x.plot(subplots=True)
forcing_data.u.plot(subplots=True)
# Load the lorenz system function for analytical derivative computation
model = lorenz_equation()
# model = lotka_volterra_equation()
# System dimensionality
dims = (sys.sim_data.shape[1]-1)//2
# Create a ProcessedSignal instance - calculate derivatives, filter out noise etc.
sig = ProcessedSignal(
    time_data, state_data.x.values, forcing_data.u.values,
    noise_power=0.1,
    spectral_cutoff=0.25,
    kernel='hann',
    kernel_size=16,
    model=model
)

# Plot derivatives comparison and SVD
plot_dxdt_comparison(sig)
# plot_svd(sig.svd)

# %%
# SINDy
dx = sig.dxdt_spectral_filtered
x = sig.x_filtered
# dx = sig.dxdt_exact
# x = sig.x_clean
theta = poly_library(x, poly_orders=(1, 2))

dx = cutoff(dx, sig.kernel_size)
x = cutoff(x, sig.kernel_size)
theta = cutoff(theta, sig.kernel_size)
u = cutoff(forcing_data.u, sig.kernel_size)
theta2 = pd.concat([theta, u], axis=1)

ksi = seq_thresh_ls(theta, dx, n=80, alpha=0.001, verbose=True, threshold=0.05)
ksi2 = seq_thresh_ls(theta2, dx, n=80, alpha=0.001, verbose=True, threshold=0.05)

fig, ax = plt.subplots(1, 2, tight_layout=True)
plot_ksi(ksi, theta, dx, ax[0], show_sparse=False)
plot_ksi(ksi2, theta2, dx, ax[1], show_sparse=False)

# %%
def create_SINDy_model(theta, ksi, thresh=0.01):
    cand_fun = theta.columns
    ksiT = ksi.T

    # Full system function string
    system_str = ''
    # Build function strings for each state function out of xi coefficients and candidate function labels
    state_fun_strings = []
    for state_fun_idx in range(ksiT.shape[1]):
        system_str += "State function x{}_dot\n".format(state_fun_idx)
        state_fun_str = ''
        for cand_fun_str, cand_fun_coeff in zip(cand_fun, ksiT[:, state_fun_idx]):
            if np.abs(cand_fun_coeff) > thresh:
                cand_str = "{c:0.5f} * {fun} + ".format(c=cand_fun_coeff, fun=cand_fun_str) # rounds to 5 decimal places
                state_fun_str += cand_str
                system_str += "\t{}\n".format(cand_str)
        state_fun_str = state_fun_str[:-3]  # cut off last 3 characters (the plus sign and two spaces)
        state_fun_strings.append(state_fun_str)
        system_str = system_str[:-3] + '\n\n'

    # Combine the state function strings into lambda output form
    lambda_str = 'lambda x, u: ['
    for state_fun_str in state_fun_strings:
        lambda_str += state_fun_str + ', '
    lambda_str = lambda_str[:-2] + ']'  # cut off last two characters and add ']'

    identified_model = eval(lambda_str)  # SINDYc identified model
    return identified_model, system_str

id_model, info = create_SINDy_model(theta2, ksi2)
print(info)

