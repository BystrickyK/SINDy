from scipy.signal import welch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath as cm
from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
from tools import mirror, halve
from containers.DynaFrame import DynaFrame
import matplotlib as mpl
import os
from definitions import ROOT_DIR

mpl.use('Qt5Agg')

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})


def mov_avg(x, size):
    conv = np.convolve(x, np.ones(size), 'same') / size
    conv[:size//2] = conv[size//2]
    conv[-size//2:] = conv[-size//2]
    return conv

class SpectralFilter:
    def __init__(self, x, dt, plot=False, mirroring=True):

        self.x = x  # data array
        self.dt = dt  # sampling period

        shape = x.shape
        if len(shape) == 1:
            self.cols = 1
        elif len(shape) == 2:
            self.cols = shape[1]
        else:
            raise ValueError

        self.plot = plot
        self.mirroring = mirroring

        self.cutoff_frequency = []
        self.cutoff_frequency_idx = []

    def find_cutoff_frequencies(self, offset=0, std_thresh=2):

        if self.plot:
            fig, axs = plt.subplots(nrows=self.x.shape[1],
                                    sharex=True,
                                    tight_layout=True)

            if not hasattr(axs, '__iter__'):
                axs = [axs]

        if not hasattr(offset, '__iter__'):
            offset = [offset]
        if not (len(offset) == len(self.x.columns)):
            offset = np.zeros_like(self.x.columns) + offset

        # for each col in x
        for i, colname in enumerate(self.x.columns):
            sig = self.x.iloc[:, i]

            # Calculate the periodogram (PSD) using Welch's method
            f, Pxx = welch(sig, fs=1. / self.dt, nperseg=512)
            f = f * 2 * np.pi  # Convert frequency from Hz to rad/s

            Pxx_smooth = mov_avg(Pxx, len(Pxx)//32)
            mean = np.mean(Pxx_smooth[int(len(Pxx)*0.5):])
            std = np.std(Pxx_smooth[int(len(Pxx)*0.5):])
            thresh = mean + std_thresh*std

            # Set the cutoff frequency as the lowest frequency at which the absolute
            # value of the derivative of PSD is below threshold
            f_cutoff_idx = int(np.where(
               Pxx_smooth < thresh
            )[0][0] + offset[i])
            f_cutoff = f[f_cutoff_idx]
            self.cutoff_frequency.append(f_cutoff)
            self.cutoff_frequency_idx.append(f_cutoff_idx)

            if self.plot:
                # if i == 0:
                #     title = 'Periodogram from Welch\'s method\n$ \hat{}_{} $'.format('x', str(i + 1))
                # else:
                #     title = '$ \hat{}_{} $'.format('x', str(i + 1))
                # axs[i].set_title(rf'{title}')
                axs[i].semilogy(f, Pxx_smooth, linestyle='none', alpha=0.7, color='tab:blue',
                                marker='o', markersize=10, label='Smoothed periodogram')
                axs[i].vlines([f_cutoff],
                              ymin=Pxx.min(), ymax=Pxx.max(),
                              linestyle=':', color='tab:grey', alpha=0.9,
                              label='Cutoff frequency')
                axs[i].hlines([thresh],
                              xmin=0, xmax=np.max(f),
                              linestyle='--', color='tab:grey', alpha=0.9,
                              label='Threshold')
                axs[i].set_ylabel(rf'$Power\ x_{i+1}$', fontsize=18)
                axs[i].legend()
                axs[i].set_xlim([-0.1*f_cutoff, 6*f_cutoff])
                axs[-1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$', fontsize=16)

        if self.plot:
            plt.show()

    def filter(self):

        x = self.x.values
        dt = self.dt

        # Calculate frequencies and Fourier coeffs
        if self.mirroring:
            x = mirror(x)

        omega, x_hat = fft(x, dt)

        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')


        if self.plot:
            fig, axs = plt.subplots(nrows=self.cols, tight_layout=True, sharex=True)

            if not hasattr(axs, '__iter__'):
                axs = [axs]

        for col in range(self.cols):
            # Find frequency index of the respective cutoff frequency
            idx_r = np.argmin(np.abs(omega - self.cutoff_frequency[col]))
            idx_l = len(omega) - idx_r

            x_hat_f[idx_l:idx_r, col] = x_hat[idx_l:idx_r, col]

            if self.plot:
                x_hat_abs = np.abs(x_hat[:, col])
                x_hat_f_abs = np.abs(x_hat_f[:, col])
                title = '$ \hat{}_{} $'.format('x', str(col+1))
                # axs[col].set_title(rf'{title}')
                axs[col].semilogy(omega, x_hat_abs, alpha=1,
                                  marker='.', linestyle='none',
                                  color='tab:blue', markersize=10,
                                  label='Discarded frequencies')
                axs[col].semilogy(omega, x_hat_f_abs, alpha=0.7,
                                  marker='.', linestyle='none',
                                  color='tab:green', markersize=10,
                                  label='Preserved frequencies')
                axs[col].vlines([omega[idx_r], omega[idx_l]], ymin=np.min(x_hat_abs), ymax=np.max(x_hat_abs),
                                linestyle=':', color='tab:grey',
                                label='Cutoff frequency')
                plt.legend()
                # axs[N-1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                # axs.set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                f_cutoff = self.cutoff_frequency[col]
                axs[col].set_xlim([-0.1*f_cutoff, 6*f_cutoff])
                axs[col].set_ylabel(r'$Power$')
                axs[-1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')

        plt.show()

        X_filtered = np.real(ifft(x_hat_f))
        if self.mirroring:
            X_filtered = halve(X_filtered)
        X_filtered = DynaFrame(X_filtered)
        X_filtered.columns = self.x.columns
        return X_filtered