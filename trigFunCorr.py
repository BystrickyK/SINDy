import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from itertools import combinations
from utils.visualization import plot_corr
import matplotlib

matplotlib.use('Qt5Agg')

filename = 'doublePendulumCart/results/simdata.csv'
sim_data = pd.read_csv(filename)

X = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1], noise_power=0)

state_data = X.x

dim = state_data.shape[1]

#%%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
angle_sums = sum_library(state_data.iloc[:, 1:dim//2], (-2, -1, 0, 1, 2))
trig_data = trigonometric_library(angle_sums)

corr = trig_data.corr()
plot_corr(corr, trig_data.columns)

corr_lower_tri = np.tril(corr, k=-1)
trig_identity_idx = np.abs(corr_lower_tri) == 1
trig_identity_idx = np.array(np.nonzero(trig_identity_idx))[0,:]
trig_identity_labels = trig_data.columns[trig_identity_idx]
trig_data_without_identities = trig_data.drop(trig_identity_labels, axis=1)

corr_no_identities = trig_data_without_identities.corr()
plot_corr(corr_no_identities, trig_data_without_identities.columns)
