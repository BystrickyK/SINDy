import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

#%%
style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
plt.style.use({'seaborn', style_path})
mpl.use('Qt5Agg')
data_path = os.path.join(ROOT_DIR, 'data', 'singlePend')

#%% Load data
path = os.path.join(data_path, 'real', 'single_mid.csv')
measurement_data = pd.read_csv(path, delimiter=';')


#%% Resample data using constant sample time 'ts'
def resample(data, ts, col_names=None):
    idxs = data.index.values



# #%% Show the time sampling differences
# diffs = np.diff(idxs)
# plt.figure()
# plt.plot(diffs)

    DataInterpolator = interp1d(idxs, data.values, axis=0)

    t_resampling = np.arange(idxs[0], idxs[-1], ts)
    data = DataInterpolator(t_resampling)
    data = pd.DataFrame(data, index=t_resampling)
    if col_names:
        data.columns = col_names
    else:
        data.columns = ['x_' + str(i + 1) for i in range(len(data.columns))]

    data.index = data.index - data.index.values[0]
    return data
