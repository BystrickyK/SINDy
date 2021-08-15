import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#%%
style_path = os.path.join(ROOT_DIR, '..', 'utils', 'visualization', 'BystrickyK.mplstyle')
plt.style.use({'seaborn', style_path})
mpl.use('Qt5Agg')
data_path = os.path.join(ROOT_DIR, '../../data', 'singlePend')

#%% Load data and create a DataFrame
# data_full = pd.read_csv('measurementData.csv', delimiter=';')
path = os.path.join(data_path, 'real', 'single_fast2.csv')
data_full = pd.read_csv(path, delimiter=';')

columns = [2, 3, 5, 7, 9, 10]
data = data_full.iloc[:, columns]
data.columns = ['t', 'x1', 'x2', 'x3', 'x4', 'u']
data['t'] = data.loc[:, 't'] - data.loc[0, 't']
data['x2'] = data.loc[:, 'x2'] / 360 * (-2) * np.pi + np.pi
# data['x2'] = data.loc[:, 'x2'] - data.loc[0, 'x2']
data['x4'] = data.loc[:, 'x4'] / 360 * (-2) * np.pi
data.set_index('t', inplace=True)
data = data.loc[87:200, :]
# data.plot(subplots=True)

#%%
labels = [
"$x_1\ [\mathrm{m}]$",
"$x_2\ [\mathrm{rad}]$",
"$x_3\ [\\frac{\mathrm{m}}{\mathrm{s}}]$",
"$x_4\ [\\frac{\mathrm{rad}}{\mathrm{s}}]$",
"$u$"
]
colors = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', 'tab:red']
fig, axs = plt.subplots(nrows=5, tight_layout=True, sharex=True)
for i, ax in enumerate(axs):
    ax.plot(data.index, data.iloc[:, i],
               color=colors[i], label=labels[i])
    ax.set_ylabel(labels[i])
axs[-1].set_xlabel("Time $t\ [\mathrm{s}]$")

#%% Show the time sampling differences
idxs = data.index.values
diffs = np.diff(idxs)
plt.figure()
plt.plot(diffs)
plt.ylabel('$\Delta^2 t$', fontsize=18)
plt.xlabel('Sample index', fontsize=16)
plt.title('Sampling time difference')

#%% Resample data using constant sample time 'ts'
ts = 0.001
DataInterpolator = interp1d(idxs, data.values, axis=0)

t_resampling = np.arange(idxs[0], idxs[-1], ts)
data = DataInterpolator(t_resampling)
data = pd.DataFrame(data, index=t_resampling)
data.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
# data.plot(subplots=True)
#%% Cut off the samples before the experiment start (when the pendulum was being mo by hand)
# data.index = data.index - data.index.values[0]
# data.plot(subplots=True, grid=True, linewidth=2)
path = os.path.join(data_path, 'real', 'processed_measurements.csv')
data.to_csv(path)