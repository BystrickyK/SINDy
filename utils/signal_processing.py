import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt


class Signal:

    def __init__(self, time_data):
        # First column must be time measurements
        self.t = time_data[:]

        # Number of samples (readings)
        self.samples = time_data.shape[0]

        # Sampling period
        self.dt = self.t[1] - self.t[0]

    def create_df(self, data, var_label='x'):
        dims = data.shape[1]
        var_labels = [var_label + '[' + str(i) + ']' for i in range(dims)]
        return pd.DataFrame(
            data=data,
            index=self.t,
            columns=var_labels
        )


class StateSignal(Signal):
    def __init__(self, time_data, state_data, noise_power=0):
        """

        Args:
            state_data (np.array): First column is time measurements, other columns are state measurements
            noise_power: How much white noise should be added to the measurements
        """
        Signal.__init__(self, time_data)

        # Signal dimensionality (number of columns)
        self.dims = state_data.shape[1]

        # DF of the original signal
        x_clean = state_data
        self.x_clean = self.create_df(x_clean)

        # The DataFrame self.x is calculated from self.x_clean via
        # the noise_power setter method
        self.x = None
        self.noise_power = noise_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        x = self.x_clean + noise_power * np.random.randn(*self.x_clean.shape)
        self.x = self.create_df(x)
        self._noise_power = noise_power


class ForcingSignal(Signal):
    def __init__(self, time_data, forcing_data):
        Signal.__init__(self, time_data)

        self.dims = forcing_data.shape[1]

        self.u = forcing_data
        self.u = self.create_df(self.u, var_label='u')

class FullSignal(Signal):
    def __init__(self, StateSignal, ForcingSignal):
        Signal.__init__(self, StateSignal.t)
        self.x = StateSignal.x
        self.u = ForcingSignal.u

class SignalProcessor(StateSignal, ForcingSignal):
    def __init__(self, time_data, state_data, forcing_data,
                 spectral_cutoff=None,
                 kernel=None,
                 kernel_size=8,
                 noise_power=0,
                 model=None):

        StateSignal.__init__(self, time_data, state_data, noise_power)
        ForcingSignal.__init__(self, time_data, forcing_data)

        self.spectral_cutoff = spectral_cutoff

        self.kernel = kernel

        self.kernel_size = kernel_size

        # Spectral derivative
        self.dxdt_spectral = self.spectral_derivative()  # Fills dxdt_spectral

        # Finite difference derivative
        self.dxdt_finitediff = self.finite_difference_derivative()  # Fills dxdt_finitediff

        self.dxdt_spectral_filtered = None
        self.x_filtered = None
        if self.kernel:
            # Filtered spectral derivative
            self.dxdt_spectral_filtered = self.convolution_filter(self.dxdt_spectral)

            # Filtering x
            self.x_filtered = self.convolution_filter(self.x)

        # Calculate exact derivative from the system model (if available)
        if self.model:
            self.model = lambda x, u: model(0, x, u)  # assume time-invariant system
            self.dxdt_exact = self.exact_derivative()

        self.svd = self.compute_svd()

    @property
    def spectral_cutoff(self):
        return self.__spectral_cutoff

    @spectral_cutoff.setter
    def spectral_cutoff(self, spectral_cutoff):
        # How many frequencies should be kept from each side of the spectrum (centered at 0 freq)
        # between (0,0.5)
        if isinstance(spectral_cutoff, float):
            spectral_cutoff = [spectral_cutoff for dim in range(self.dims)]
        self.__spectral_cutoff = spectral_cutoff

    @property
    def kernel(self):
        return self.__kernel

    @kernel.setter
    def kernel(self, kernel):
        # Kernel filtering
        self.__kernel = kernel  # Which window should be used (from scipy.signals.windows)

    @property
    def kernel_size(self):
        return self.__kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        self.__kernel_size = kernel_size  # Size of the window

    # Calculates the spectral derivative from self.x
    def spectral_derivative(self):
        L = self.t[-1]  # Domain length (~ total time)
        n = self.samples

        # Fourier coefficients
        x_hat = np.fft.fft(self.x, axis=0)
        x_hat = np.fft.fftshift(x_hat, axes=0)

        # Fourier frequencies
        omega = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
        omega = omega[:, np.newaxis]

        if self.spectral_cutoff:
            # Create mask
            mask = np.zeros([self.samples, self.dims])
            for idx, cutoff in enumerate(self.spectral_cutoff):
                cutoff = self.samples / 2 - cutoff * self.samples
                cutoff = int(cutoff)
                mask[cutoff: -cutoff, idx] = 1
            # Truncate Fourier coefficients
            x_hat = x_hat * mask

        dxdt_hat = 1j * omega * x_hat
        dxdt_hat = np.fft.ifftshift(dxdt_hat, axes=0)
        dxdt = np.real(np.fft.ifft(dxdt_hat, axis=0))
        return self.create_df(dxdt)

    # Convolution filtering
    def convolution_filter(self, x):
        if self.kernel == 'hann':
            krnl = scipy.signal.hann(self.kernel_size)
        elif self.kernel == 'flattop':
            krnl = scipy.signal.flattop(self.kernel_size)
        else:
            return

        krnl /= sum(krnl)  # Normalize kernel

        x_filtered = np.apply_along_axis(
            lambda col: scipy.signal.convolve(col, krnl, mode='valid'),
            0, x)
            # lambda col: scipy.signal.sosfiltfilt(krnl, col, mode='same'),
            # 0, x)
        return self.create_df(x_filtered)

    # Calculates the finite difference derivative
    def finite_difference_derivative(self, direction='forward'):
        if direction == 'forward':
            dxdt = (np.diff(self.x, axis=0)) / self.dt  # last value is missing
            dxdt = np.vstack((dxdt, dxdt[-1, :]))
            return self.create_df(dxdt)
        elif direction == 'backward':
            x = np.flip(self.x.values, axis=0)
            dxdt = (-np.diff(x, axis=0)) / self.dt
            dxdt = np.flip(dxdt, axis=0)  # first value is missing
            dxdt = np.vstack((dxdt[0, :], dxdt))
            return self.create_df(dxdt)

    def exact_derivative(self):
        dxdt = np.array([*map(self.model, self.x_clean.values, self.u.values)])
        return self.create_df(dxdt)

    def compute_svd(self):
        u, s, vt = np.linalg.svd(self.x.values.T, full_matrices=False)
        s = np.diag(s)
        svd = {'U': u, 'Sigma': s, 'V*': vt}
        return svd
