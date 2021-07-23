import pandas as pd
import matplotlib.pyplot as plt
from src.utils.function_libraries import *
from src.utils.data_utils import *
from src.utils.identification.PI_Identifier import PI_Identifier
from src.utils.solution_processing import *
from differentiation.spectral_derivative import compute_spectral_derivative
from differentiation.finite_diff import compute_finite_diff
from filtering.SpectralFilter import SpectralFilter
from filtering.KernelFilter import KernelFilter
from tools import halve, mirror, add_noise, downsample
from src.utils.theta_processing.single_pend import *
from sklearn.model_selection import TimeSeriesSplit
import matplotlib as mpl
import os
import pickle
from containers.DynaFrame import DynaFrame, create_df
from definitions import ROOT_DIR
import sympy as sp
from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})

mpl.use('Qt5Agg')

datafile = 'singlePend.csv'
data_path = os.path.join(ROOT_DIR,'data','singlePend','simulated',datafile)
cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')

# Get training dataset
sim_data = pd.read_csv(data_path)
dt = sim_data.loc[1, 't'] - sim_data.loc[0, 't']
X = sim_data.loc[:, ('x1', 'x2')]
DX = sim_data.loc[:, ('dx1', 'dx2')]

#%%
# steps = np.arange(1,80,1)
# errs_spectral = []
# errs_findiff = []
# for step in steps:
#     x = downsample(X, step)
x = X
dx = DX
#     dx = downsample(DX, step)
# dx_from_clean_spectral = compute_spectral_derivative(x, dt, mirroring=True)
# dx_from_clean_findiff = compute_finite_diff(x, dt)
# compare_signals3(dx, dx_from_clean_spectral, dx_from_clean_findiff, ['Real', 'Spectral', 'Finite Difference'],
#                  ylabels=['$\dot{x}_1 \; [m\; s^{-1}]$', '$\dot{x}_2 \; [rad\; s^{-1}]$'],
#                 title_str='Spectral derivative from clean data')
# plt.xlim([5000, 8000])


std = 0.05
noise = np.random.randn(*x.shape) * std
x_noisy = x + noise
dx_from_noisy_spectral = compute_spectral_derivative(x_noisy, dt, mirroring=True)
dx_from_noisy_findiff = compute_finite_diff(x_noisy, dt)
compare_signals3(dx, dx_from_noisy_spectral, dx_from_noisy_findiff, ['Real', 'Spectral', 'Finite Difference'],
                 ylabels=['$\dot{x}_1 \; [m\; s^{-1}]$','$\dot{x}_2 \; [rad\; s^{-1}]$'],
                title_str='Spectral derivative from slightly noisy data')
plt.xlim([5000, 8000])

compare_signals(x, x_noisy, ['Clean', 'Noisy'],
                ylabels=['$x_1 \; [m]$', '$x_2 \; [rad]$'],
                 title_str='')
plt.xlim([5000, 8000])

filter = SpectralFilter(x_noisy, dt)
filter.find_cutoff_frequencies()
x_filt_spec = filter.filter()
filter = KernelFilter(kernel_size=101)
x_filt_kern = filter.filter(x_noisy)
dx_from_filt_spectral_kernelfilt = compute_spectral_derivative(x_filt_kern, dt, mirroring=True)
dx_from_filt_spectral_specfilt = compute_spectral_derivative(x_filt_spec, dt, mirroring=True)
compare_signals3(dx, dx_from_filt_spectral_kernelfilt, dx_from_filt_spectral_specfilt,
                 ['Real', 'Kernel filtered', 'Spectral filtered'],
                 ylabels=['$\dot{x}_1 \; [m\; s^{-1}]$','$\dot{x}_2 \; [rad\; s^{-1}]$'],
                 title_str='Spectral derivative from filtered data')

    # dx_ = dx.iloc[20:-20]
    # dx_from_clean_spectral_ = dx_from_clean_spectral[20:-20]
    # dx_from_clean_findiff_ = dx_from_clean_findiff[20:-20]
    # err_spectral = dx_ - dx_from_clean_spectral_
    # err_findiff = dx_ - dx_from_clean_findiff_
    # # compare_signals(err_spectral, err_findiff, ['Spectral', 'Finite Difference'],
    # #                 ylabels=['$\dot{x}_1 \; [m\; s^{-1}]$', '$\dot{x}_2 \; [rad\; s^{-1}]$'],
    # #                 title_str='Error')
    # errs_spectral.append(np.sum(np.square(err_spectral)))
    # errs_findiff.append(np.sum(np.square(err_findiff)))

# errs_spectral = np.array(errs_spectral)[::-1]
# errs_findiff = np.array(errs_findiff)[::-1]
# steps = np.array(steps)[::-1] * dt
# fig, ax = plt.subplots(nrows=2, tight_layout=True, figsize=(10, 6))
# for i in (0,1):
#     ax[i].plot(steps, errs_spectral[:,i], color='tab:red')
#     ax[i].plot(steps, errs_findiff[:,i], color='tab:blue')
#     ax[i].set_yscale('log')
#     ax[i].legend(['Spectral', 'Finite Difference'])
#
# ax[0].set_title("Squared error as a function of sampling period")
# ax[0].set_ylabel('$\dot{x}_1$ error')
# ax[1].set_ylabel('$\dot{x}_2$ error')
# ax[1].set_xlabel('Sampling period $\Delta t$')

