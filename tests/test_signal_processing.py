from src.utils.data_utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as ss
import os

# os.chdir('..')
# print(os.getcwd())
# dirname = os.getcwd() + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
# filename = dirname + 'simdata_slow.csv'
# data = pd.read_csv(filename)
# data = data.iloc[2000:6000, :].reset_index(drop=True)

t = np.linspace(0, 6*np.pi, 20000)
data = np.array([t, np.sin(t)+np.sin(5*t), np.cos(t)**2+np.cos(8*t)**2, ss.square(t)]).T
data, dt = remove_time(data)
data = create_df(data, 'x')
# data.plot(subplots=True, title='Clean data')

data_diff = create_df(compute_spectral_derivative(data, dt), 'x')
data_diff.plot(subplots=True, title='Clean spectral differentiated data')

datan = add_noise(data, [0.1, 2*np.pi*0.01, 0.1])
# datan.plot(subplots=True, title='Noisy data')

filter = SpectralFilter(datan, dt, plot=True)
filter.find_cutoffs(0.9, 1)
dataf = filter.filter()
# dataf.plot(subplots=True, title='Spectral filtered noisy data')

# filter2 = KernelFilter(kernel='hann', kernel_size=80)
# dataf2 = filter2.filter(datan)

plotdata = [data, datan, dataf]
with plt.style.context('seaborn'):
    fig, axs = plt.subplots(nrows=data.shape[1], ncols=1, tight_layout=True)
    for i, ax in enumerate(axs):
        ax.plot(plotdata[0].iloc[:, i], alpha=0.8)
        ax.plot(plotdata[1].iloc[:, i], alpha=0.6)
        ax.plot(plotdata[2].iloc[:, i], alpha=0.8)
        # ax.plot(t, plotdata[3].iloc[:, i], alpha=0.8)
        ax.legend(['Clean', 'Noisy', 'SFiltered', 'KFiltered'])
