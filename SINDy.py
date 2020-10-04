from LorenzAnimation import AnimatedLorenz
from equations.LorenzEquation import lorenz_equation_p
import numpy as np
import matplotlib.pyplot as plt
from signal_processing import ProcessedSignal
from function_library import function_library
from sklearn import linear_model

def plot_ksi(ksi, theta, dx):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 4)

    maxabs = np.abs(ksi).max()
    p = ax.matshow(ksi.T, cmap='PRGn', vmin=-maxabs/5, vmax=maxabs/5)
    plt.colorbar(p, ax=ax)
    ax.set_yticks([*range(theta.shape[1])])
    ax.set_yticklabels(theta.columns)
    ax.set_ylabel("Candidate functions")
    ax.set_xticks([*range(dx.shape[1])])
    ax.set_xticklabels(dx.columns)
    ax.set_xlabel("dx/dt")
    ax.set_title("Ksi")

tmax = 30
x0 = [40, -40, -15]
sys = AnimatedLorenz(x0, tmax, anim_speed=30, dt=0.0025)

lorenz = lorenz_equation_p()
lorenz_ti = lambda x: lorenz(0, x)
sig = ProcessedSignal(
    sys.sim_data,
    noise_power=0.1,
    spectral_cutoff=[3000, 3000, 3000],
    kernel='hann',
    kernel_size=32,
    model=lorenz_ti)

sig.plot_dxdt_comparison()
sig.plot_svd()

# SINDy
dx = sig.dxdt_spectral_filtered
theta = function_library(sig, poly_orders=(1,2,3))
lasso_model_spectral = linear_model.Lasso(max_iter=25000)
lasso_model_spectral.fit(theta.values, dx.values)
ksi_spectral = lasso_model_spectral.coef_
plot_ksi(ksi_spectral, theta, dx)
plt.title("Spectral")
# SINDy
dx = sig.dxdt_exact
theta = function_library(sig, poly_orders=(1,2,3))
lasso_model = linear_model.Lasso(max_iter=25000)
lasso_model.fit(theta.values, dx.values)
ksi = lasso_model.coef_
plot_ksi(ksi, theta, dx)
plt.title("Exact")
