import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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
