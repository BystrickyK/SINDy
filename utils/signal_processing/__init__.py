import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import cmath as cm

def remove_time(data):
    # Removes the time column and returns the time period and the data

    if isinstance(data, pd.DataFrame):
        dt = data.iloc[1, 0] - data.iloc[0, 0]
        data = data.iloc[:, 1:]

    if isinstance(data, np.ndarray):
        dt = data[1, 0] - data[0, 0]
        data = data[:, 1:]

    return data, dt

def energy_normalize(data):
    col_energy = np.square(data).sum(axis=0)  # sum squares over the rows
    energy_normalized_data = data / np.sqrt(col_energy)  # divide every column by the sqrt of the respective energy
    energy_normalized_data *= 1e7  # set the column's energy (1e0 would be too low)
    return energy_normalized_data

def cutoff_ends(data, cutoff):
    return data.iloc[cutoff:-cutoff, :]

def fft(x, dt):
    n = x.shape[0]  # Number of samples
    fs = 1/dt  # Sampling freq

    # Fourier coefficients
    x_hat = np.fft.fft(x, axis=0)
    x_hat = np.fft.fftshift(x_hat, axes=0)

    # Fourier frequencies in rad/s
    omega = (fs*2*np.pi/n) * np.arange(-n / 2, n / 2)
    omega = omega[:, np.newaxis]

    return omega, x_hat

def ifft(x_hat):
    x_hat = np.fft.ifftshift(x_hat, axes=0)
    x = np.real(np.fft.ifft(x_hat, axis=0))
    return x

# Calculates the spectral derivative from x
def compute_spectral_derivative(x, dt):
    """
    x (DataFrame): State measurements
    dt (Float): Time step size
    """
    if isinstance(x, pd.DataFrame):
        x = np.array(x)

    omega, x_hat = fft(x, dt)

    dxdt_hat = 1j * omega * x_hat
    dxdt_hat = np.fft.ifftshift(dxdt_hat, axes=0)
    dxdt = np.real(np.fft.ifft(dxdt_hat, axis=0))

    return dxdt

def create_df(data, var_label='x'):
    try:
        dims = data.shape[1]
    except IndexError:
        dims = 1

    df = pd.DataFrame(data)

    if dims > 1:
        var_labels = [var_label + '_' + str(i+1) for i in range(dims)]
        df.columns = var_labels

    return df

def add_noise(data, noise_power):
    additive_noise = noise_power * np.random.randn(*data.shape)
    data_noisy = data + additive_noise
    data_noisy = create_df(data_noisy)
    return data_noisy

class SpectralFilter:
    def __init__(self, X, dt, plot=False):
        #

        self.X = X
        self.dt = dt

        shape = X.shape
        print(shape)
        print(len(shape))
        if len(shape) == 1:
            self.cols = 1
        elif len(shape) == 2:
            self.cols = shape[1]
        else:
            raise ValueError

        self.plot = plot

        self.cutoffs = []

    def find_cutoffs_and_meanlogpower(self, k=0.95, freq_multiplier=1):

        if self.plot:
            fig, axs = plt.subplots(nrows=self.X.shape[1], sharex=True, tight_layout=True)

        # for each col in X
        for i in range(self.cols):
            sig = self.X.iloc[:, i]

            # Calculate the periodogram (PSD) using Welch's method
            f, Pxx = welch(sig, fs=1. / self.dt, nperseg=1024)
            f = f * 2 * np.pi  # Convert frequency from Hz to rad/s

            # Find the cutoff frequence
            Pxx_meanlog = np.exp(k * np.mean(np.log(Pxx)))
            f_cutoff_idx = np.where(np.diff(np.sign(np.log(Pxx) - np.log(Pxx_meanlog))))[-1][-1] + 1
            f_cutoff = f[f_cutoff_idx] * freq_multiplier  # Multiply the cutoff frequency
            self.cutoffs.append(f_cutoff)
            self.meanlogpower = Pxx_meanlog

            if self.plot:
                if i == 0:
                    title = 'Periodogram from Welch\'s method\n$ \hat{}_{} $'.format('x', str(i + 1))
                else:
                    title = '$ \hat{}_{} $'.format('x', str(i + 1))
                axs[i].set_title(rf'{title}')
                axs[i].semilogy(f, Pxx, linewidth=3, alpha=0.9, marker='o')
                axs[i].hlines(Pxx_meanlog, xmin=0, xmax=f[-1],
                              linestyle=':', color='black',
                              alpha=0.7)
                axs[i].vlines([f_cutoff],
                              ymin=Pxx.min(), ymax=Pxx.max(),
                              linestyle=':', color='black', alpha=0.9)
                axs[i].vlines([f_cutoff / freq_multiplier],
                              ymin=Pxx.min(), ymax=Pxx.max(),
                              linestyle='-.', color='black', alpha=0.9)
                axs[i].set_ylabel(r'$Power$')
                axs[self.N - 1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')

        if self.plot:
            plt.show()

    def filter(self, var_label='x'):

        x = self.X.values
        dt = self.dt

        # Calculate frequencies and Fourier coeffs
        omega, x_hat = fft(x, dt)

        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')

        shape = x.shape
        print(shape)
        print(len(shape))
        if len(shape) == 1:
            N = 1
        elif len(shape) == 2:
            N = shape[1]
        else:
            raise ValueError
        if self.plot:
            fig, axs = plt.subplots(nrows=N, tight_layout=True, sharex=True)

        for col in range(N):
            # Find frequency index of the respective cutoff frequency
            idx_r = np.argmin(np.abs(omega - self.cutoffs[col]))
            idx_l = len(omega) - idx_r

            x_hat_f[idx_l:idx_r, col] = x_hat[idx_l:idx_r, col]

            if self.plot:
                x_hat_abs = np.abs(x_hat[:, col])
                x_hat_f_abs = np.abs(x_hat_f[:, col])
                title = '$ \hat{}_{} $'.format('x', str(col+1))
                axs[col].set_title(rf'{title}')
                axs[col].semilogy(omega, x_hat_abs, alpha=1,
                                  marker='.', linestyle='none')
                axs[col].semilogy(omega, x_hat_f_abs, alpha=0.7,
                                  marker='.', linestyle='none')
                axs[col].vlines([omega[idx_r], omega[idx_l]], ymin=np.min(x_hat_abs), ymax=np.max(x_hat_abs),
                                linestyle=':', color='black')
                axs[N-1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
                axs[col].set_ylabel(r'$Power$')

                plt.show()

        self.X_filtered = create_df(ifft(x_hat_f), var_label=var_label)
        return self.X_filtered

    def subtract_meanpower_from_modulus(self, x=None, dt=None, var_label='x'):
        if x is None:
            x = self.X.values
            dt = self.dt

        # Calculate frequencies and Fourier coeffs
        omega, x_hat = fft(x, dt)

        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')

        N = x.shape[1] # Number of signals
        for col in range(N):
            x_hat_polar = np.array([cm.polar(x_h) for x_h in x_hat[:, col]])
            mean_rad = np.exp(self.meanlogpower)  # Mean radius
            x_hat_polar[:, 0] = x_hat_polar[:, 0] - mean_rad
            x_hat_f_polar = np.empty_like(x_hat_polar)
            for xh in x_hat_polar:
                x_hat_f[:, col] = [cm.rect(rad, phs) for rad,phs in x_hat_f_polar]

        self.X_filtered = create_df(ifft(x_hat_f), var_label=var_label)
        return self.X_filtered

    # Should remove higher frequency peaks in the PSD
    # not very reliable
    def remove_peaks(self, x=None, var_label='x', plot=True):
        if x is None:
            x = self.X.values.values
            dt = self.X.dt


        # Calculate frequencies and Fourier coeffs
        omega, x_hat = fft(x, dt)
        x_hat_real = np.real(x_hat)
        filt = KernelFilter(kernel_size=50)
        x_hat_real = filt.filter(x_hat_real).values

        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')

        N = x.shape[1] # Number of signals
        L = x.shape[0] # Number of samples


        if plot:
            fig, axs = plt.subplots(nrows=N, tight_layout=True, sharex=True)

        for col in range(N):
            # Find frequency index of the respective cutoff frequency
            [peaks, peak_props] = find_peaks(x_hat_real[:, col], prominence=1000, distance=L/200)
            if len(peaks)%2 == 1 and len(peaks)>1:  # If # of peaks is odd and >1
                idx_l = peaks[0]  # Pick the outer peaks
                idx_r = peaks[-1]
                x_hat[idx_l-50:idx_l+50, col] = 0  # And set the neighborhoods to 0
                x_hat[idx_r-50:idx_r+50, col] = 0
                if plot:
                    axs[col].vlines([idx_l, idx_r], ymin=-1000, ymax=1000,
                                    color='red', linewidth=3)

            if plot:
                axs[col].plot(x_hat_real[:, col], '.')

        self.X_filtered = create_df(ifft(x_hat), var_label=var_label)
        return self.X_filtered


class KernelFilter:
    def __init__(self, kernel='hann', kernel_size=8):
        self.kernel = kernel
        self.kernel_size = kernel_size

    # Convolution filtering
    def filter(self, x, var_label='x'):
        if self.kernel == 'hann':
            krnl = scipy.signal.hann(self.kernel_size)
        elif self.kernel == 'flattop':
            krnl = scipy.signal.flattop(self.kernel_size)
        else:
            return

        krnl /= sum(krnl)  # Normalize kernel

        x_filtered = np.apply_along_axis(
            lambda col: scipy.signal.convolve(col, krnl, mode='same'),
            0, x)
        return create_df(x_filtered, var_label=var_label)

def compute_finite_differences(x, dt, direction='forward'):
    """
    x (DataFrame): State measurements
    dt (Float): Time step size
    """
    if direction == 'forward':
        dxdt = (np.diff(x, axis=0)) / dt  # last value is missing
        dxdt = np.vstack((dxdt, dxdt[-1, :]))
        return dxdt
    elif direction == 'backward':
        x = np.flip(x.values, axis=0)
        dxdt = (-np.diff(x, axis=0)) / dt
        dxdt = np.flip(dxdt, axis=0)  # first value is missing
        dxdt = np.vstack((dxdt[0, :], dxdt))
        return dxdt
