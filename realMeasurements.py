import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%% Load data and create a DataFrame
data_full = pd.read_csv('measurementData.csv', delimiter=';')

columns = [2, 3, 5, 7, 9]
data = data_full.iloc[:, columns]
data.columns = ['t', 'x1', 'x2', 'x3', 'x4']
data['t'] = data['t'] - data['t'][0]
data.set_index('t', inplace=True)
# data.plot(subplots=True)

#%% Show the time sampling differences
idxs = data.index.values
diffs = np.diff(idxs)
plt.figure()
plt.plot(diffs)

#%% Resample data using constant sample time 'ts'
ts = 0.001
DataInterpolator = interp1d(idxs, data.values, axis=0)

t_resampling = np.arange(idxs[0], idxs[-1], ts)
data = DataInterpolator(t_resampling)
data = pd.DataFrame(data, index=t_resampling)
data.columns = ['x_1', 'x_2', 'x_3', 'x_4']
# data.plot(subplots=True)
#%% Cut off the samples before the experiment start (when the pendulum was being mo by hand)
data = data.loc[8.5:, :]
# data.index = data.index - data.index.values[0]
data.plot(subplots=True, grid=True, linewidth=2)
