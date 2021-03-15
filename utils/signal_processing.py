import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt


def create_df(data, var_label='x'):
    try:
        dims = data.shape[1]
    except IndexError:
        dims = 1

    df = pd.DataFrame(data)

    if dims > 1:
        var_labels = [var_label + '[' + str(i+1) + ']' for i in range(dims)]
        df.columns = var_labels

    return df


class Signal:

    def __init__(self, time_data):
        # First column must be time measurements
        self.t = time_data[:]

        # Number of samples (readings)
        self.samples = time_data.shape[0]

        # Sampling period
        self.dt = self.t[1] - self.t[0]




class StateSignal(Signal):
    def __init__(self, time_data, state_data, relative_noise_power=0):
        """

        Args:
            state_data (np.array): First column is time measurements, other columns are state measurements
            relative_noise_power: How much white noise should be added to the measurements. A relative noise
                power of 0.1 means that the stdev of the additive white noise for each signal will be 10% of
                the signal's stdev.
        """
        Signal.__init__(self, time_data)

        # Signal dimensionality (number of columns)
        self.dims = state_data.shape[1]

        # DF of the original signal
        self.values_clean = create_df(state_data)

        # The DataFrame self.x is calculated from self.x_clean via
        # the noise_power setter method
        self.values = None
        self.noise_power = relative_noise_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        state_signal_powers = self.values_clean.std()
        additive_noise = np.vstack(noise_power * state_signal_powers).T * np.random.randn(*self.values_clean.shape)
        x = self.values_clean + additive_noise
        self.x = create_df(x)
        self._noise_power = noise_power

class StateDerivativeSignal(Signal):
    def __init__(self, state_signal, method='spectral'):
        """

        Args:
            state_data (np.array): First column is time measurements, other columns are state measurements
        """
        Signal.__init__(self, state_signal.t)

        # Signal dimensionality (number of columns)
        self.dims = state_signal.x.shape[1]

        # The DataFrame self.dx is calculated via numerical differentiation
        if method=='spectral':
            Differentiator = SpectralDifferentiator()
            self.values = Differentiator.compute_derivative(state_signal, self.dt)
            Filter = KernelFilter(kernel='flattop', kernel_size=5)
            self.values = Filter.filter(self.values, var_label='dx')

class ForcingSignal(Signal):
    def __init__(self, time_data, forcing_data):
        Signal.__init__(self, time_data)

        try:
            self.dims = forcing_data.shape[1]
        except IndexError:
            self.dims = 1

        self.values = create_df(forcing_data, var_label='u')

class KernelFilter:
    def __init__(self, kernel='hann', kernel_size=8):
        self.kernel = kernel
        self.kernel_size = kernel_size

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

class SpectralDifferentiator:
    def __init__(self, spectral_cutoff=None):
        self.spectral_cutoff = spectral_cutoff

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

    # Calculates the spectral derivative from self.x
    def compute_derivative(self, x, dt, var_label='dx'):
        """
        x (DataFrame): State measurements
        dt (Float): Time step size
        """
        n = x.shape[0]  # Number of samples
        dims = x.shape[1]
        L = n*dt  # Time domain length

        # Fourier coefficients
        x_hat = np.fft.fft(self.x, axis=0)
        x_hat = np.fft.fftshift(x_hat, axes=0)

        # Fourier frequencies
        omega = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
        omega = omega[:, np.newaxis]

        if self.spectral_cutoff:
            # Create mask
            mask = np.zeros([n, dims])
            for idx, cutoff in enumerate(self.spectral_cutoff):
                cutoff = n / 2 - cutoff * n
                cutoff = int(cutoff)
                mask[cutoff: -cutoff, idx] = 1  # Keep coeffs between cutoff indices
            # Truncate Fourier coefficients
            x_hat = x_hat * mask

        dxdt_hat = 1j * omega * x_hat
        dxdt_hat = np.fft.ifftshift(dxdt_hat, axes=0)
        dxdt = np.real(np.fft.ifft(dxdt_hat, axis=0))
        return create_df(dxdt, var_label=var_label)



class FiniteDifferentiator:
    def __init__(self):
        """
        TODO: Class hasn't been tested after code refactoring
        """
        pass

    # Calculates the finite difference derivative
    def compute_derivative(self, x, dt, var_label='dx', direction='forward'):
        """
        x (DataFrame): State measurements
        dt (Float): Time step size
        """
        if direction == 'forward':
            dxdt = (np.diff(x, axis=0)) / dt  # last value is missing
            dxdt = np.vstack((dxdt, dxdt[-1, :]))
            return create_df(dxdt, 'dx')
        elif direction == 'backward':
            x = np.flip(x.values, axis=0)
            dxdt = (-np.diff(x, axis=0)) / dt
            dxdt = np.flip(dxdt, axis=0)  # first value is missing
            dxdt = np.vstack((dxdt[0, :], dxdt))
            return create_df(dxdt, var_label=var_label)

class ModelDerivative:
    def __init__(self, model):
        """
        TODO: Class hasn't been tested after code refactoring
        """
        self.model = lambda x, u: model(0, x, u)  # assume time-invariant system

    def exact_derivative(self, x, u, var_label='dx'):
        """
        x (DataFrame): State measurements
        u (DataFrame): System inputs
        """
        dxdt = np.array([*map(self.model, x.values, u.values)])
        return create_df(dxdt, var_label=var_label)

