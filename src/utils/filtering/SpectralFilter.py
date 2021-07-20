from scipy.signal import welch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath as cm
from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft

class SpectralFilter:
    def __init__(self, x, dt, plot=False):

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

        self.cutoff_frequency = []
        self.cutoff_frequency_idx = []

    def find_cutoff_frequencies(self):

        if self.plot:
            fig, axs = plt.subplots(nrows=self.x.shape[1],
                                    sharex=True,
                                    tight_layout=True)

        # for each col in x
        for i, colname in enumerate(self.x.columns):
            sig = self.x.iloc[:, i]

            # Calculate the periodogram (PSD) using Welch's method
            f, Pxx = welch(sig, fs=1. / self.dt, nperseg=256)
            f = f * 2 * np.pi  # Convert frequency from Hz to rad/s

            dPxx_df = np.diff(Pxx, append=0)
            dPxx_df_logabs = np.log(np.abs(dPxx_df))

            # Set the cutoff frequency as the lowest frequency at which the absolute
            # value of the derivative of PSD is below threshold
            thresh = np.min(dPxx_df_logabs[:-2]) # ignore last element
            thresh += 6

            f_cutoff_idx = np.where(
               dPxx_df_logabs < thresh
            )[0][0]
            f_cutoff = f[f_cutoff_idx]
            self.cutoff_frequency.append(f_cutoff)
            self.cutoff_frequency_idx.append(f_cutoff_idx)

            if self.plot:
                if i == 0:
                    title = 'Periodogram from Welch\'s method\n$ \hat{}_{} $'.format('x', str(i + 1))
                else:
                    title = '$ \hat{}_{} $'.format('x', str(i + 1))
                axs.set_title(rf'{title}')
                axs.semilogy(f, Pxx, linestyle='none', alpha=0.8, marker='o', markersize=10)
                axs.vlines([f_cutoff],
                              ymin=Pxx.min(), ymax=Pxx.max(),
                              linestyle=':', color='black', alpha=0.9)
                axs.set_ylabel(r'$Power$')
                axs.set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                # axs[self.x.shape[1] - 1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')

        if self.plot:
            plt.show()


    def filter(self):

        x = self.x.values
        dt = self.dt

        # Calculate frequencies and Fourier coeffs
        omega, x_hat = fft(x, dt)

        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')


        if self.plot:
            fig, axs = plt.subplots(nrows=self.cols, tight_layout=True, sharex=True)

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
                axs.semilogy(omega, x_hat_abs, alpha=1,
                                  marker='.', linestyle='none')
                axs.semilogy(omega, x_hat_f_abs, alpha=0.7,
                                  marker='.', linestyle='none')
                axs.vlines([omega[idx_r], omega[idx_l]], ymin=np.min(x_hat_abs), ymax=np.max(x_hat_abs),
                                linestyle=':', color='black')
                # axs[N-1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                axs.set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                axs.set_ylabel(r'$Power$')

                plt.show()

        X_filtered = ifft(x_hat_f)
        return X_filtered