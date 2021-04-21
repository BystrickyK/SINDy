import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

def energy_normalize(data):
    col_energy = np.square(data).sum(axis=0)  # sum squares over the rows
    energy_normalized_data = data / np.sqrt(col_energy)  # divide every column by the sqrt of the respective energy
    # energy_normalized_data *= 1e7  # set the column's energy (1e0 would be too low)
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



class Signal:

    def __init__(self, dt):
        self.dt = dt


class StateSignal(Signal):
    def __init__(self, state_data, dt, noise_power=0, relative_noise_power=0):
        """

        Args:
            state_data (np.array): First column is time measurements, other columns are state measurements
            relative_noise_power: How much white noise should be added to the measurements. A relative noise
                power of 0.1 means that the stdev of the additive white noise for each signal will be 10% of
                the signal's stdev.
        """
        Signal.__init__(self, dt)

        # Signal dimensionality (number of columns)
        self.dims = state_data.shape[1]

        # DF of the original signal
        self.values_clean = create_df(state_data)

        # The DataFrame self.x is calculated from self.x_clean via
        # the noise_power setter method
        self.values = None
        self.relative_noise_power = relative_noise_power

    @property
    def relative_noise_power(self):
        return self._relative_noise_power

    @relative_noise_power.setter
    def relative_noise_power(self, noise_power):
        state_signal_powers = self.values_clean.std()
        additive_noise = np.vstack(noise_power * state_signal_powers).T * np.random.randn(*self.values_clean.shape)
        x = self.values_clean + additive_noise
        self.values = create_df(x)
        self._relative_noise_power = noise_power

class StateDerivativeSignal(Signal):
    def __init__(self, state_signal, method='spectral', kernel_size=11, ):
        """

        Args:
            state_data (np.array): First column is time measurements, other columns are state measurements
        """
        Signal.__init__(self, state_signal.dt)

        # Signal dimensionality (number of columns)
        self.dims = state_signal.values.shape[1]

        # The DataFrame self.dx is calculated via numerical differentiation
        if method=='spectral':
            Differentiator = SpectralDifferentiator()
            self.values = Differentiator.compute_derivative(state_signal.values, self.dt)
            Filter = KernelFilter(kernel='hann', kernel_size=kernel_size)
            self.values = Filter.filter(self.values, var_label='dx')

        if method=='finitediff':
            Differentiator = FiniteDifferentiator()
            self.values = Differentiator.compute_derivative(state_signal.values, self.dt)


class ForcingSignal(Signal):
    def __init__(self, forcing_data, dt):
        Signal.__init__(self, dt)

        try:
            self.dims = forcing_data.shape[1]
        except IndexError:
            self.dims = 1

        self.values = create_df(forcing_data, var_label='u')

class SpectralFilter:
    def __init__(self, X,  plot=False):

        self.X = X
        self.dt = X.dt
        self.N = X.values.shape[1]

        self.cutoffs = []


    def find_cutoffs(self, k=0.8, gamma=1.5, plot=True):

        self.cutoffs = []

        if plot:
            fig, axs = plt.subplots(nrows=self.N, sharex=True, tight_layout=True)
        for i in range(self.N):

            f, Pxx = welch(self.X.values.values[:, i], fs=1. / self.dt, nperseg=1024)
            f = f * 2 * np.pi  # Convert from Hz to rad/s
            Pxx_meanlog = np.exp(k * np.mean(np.log(Pxx)))
            # Pxx_diff = np.exp(np.diff(np.log(Pxx)))
            f_cutoff_idx = np.where(np.diff(np.sign(np.log(Pxx) - np.log(Pxx_meanlog))))[-1] + 1
            f_cutoff = f[f_cutoff_idx] * gamma  # Multiply the cutoff frequency
            self.cutoffs.append({'idx': f_cutoff_idx, 'cutoff': f_cutoff})
            # f_f, Pxx_f = welch(X.values.values[:, i], 1/dt)

            if plot:
                if i == 0:
                    title = 'Power Spectral Density from Welch\'s method\n$ \hat{}_{} $'.format('x', str(i + 1))
                else:
                    title = '$ \hat{}_{} $'.format('x', str(i + 1))
                axs[i].set_title(rf'{title}')
                axs[i].semilogy(f, Pxx, linewidth=3, alpha=0.8, marker='o')
                # axs[i].semilogy(f[:-1], Pxx_diff, linewidth=2, alpha=0.7)

                axs[i].hlines(Pxx_meanlog, xmin=0, xmax=f[-1],
                              linestyle=':', color='black')
                axs[i].vlines([f_cutoff, f_cutoff / gamma], ymin=Pxx.min(), ymax=Pxx.max(),
                              linestyle='-.', color='black')
                # axs[i].semilogy(f_f, Pxx_f, linewidth=2, alpha=0.7)
                axs[i].set_ylabel(r'$PSD$')
                axs[self.N - 1].set_xlabel(r'$Frequency \quad [\frac{rad}{s}]$')
        if plot:
            plt.show()

    def filter(self, x=None, dt=None, var_label='x', plot=True):
        if x is None:
            x = self.X.values.values
            dt = self.X.dt


        # Calculate frequencies and Fourier coeffs
        omega, x_hat = fft(x, dt)
        # Initialize array for filtered data in Fourier domain
        x_hat_f = np.zeros_like(x, dtype='complex')

        N = x.shape[1] # Number of signals
        if plot:
            fig, axs = plt.subplots(nrows=N, tight_layout=True, sharex=True)
        for col in range(N):
            # Find frequency index of the respective cutoff frequency
            idx_r = np.argmin(np.abs(omega - self.cutoffs[col]['cutoff']))
            idx_l = len(omega) - idx_r

            x_hat_f[idx_l:idx_r, col] = x_hat[idx_l:idx_r, col]

            if plot:
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
                axs[col].set_ylabel(r'$PSD$')

                plt.show()

        self.X_filtered = create_df(ifft(x_hat_f), var_label=var_label)
        return self.X_filtered

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
    def __init__(self):
        pass

    # Calculates the spectral derivative from self.x
    def compute_derivative(self, x, dt, var_label='dx'):
        """
        x (DataFrame): State measurements
        dt (Float): Time step size
        """
        n = x.shape[0]  # Number of samples
        dims = x.shape[1]

        # # Fourier coefficients
        # x_hat = np.fft.fft(x, axis=0)
        # x_hat = np.fft.fftshift(x_hat, axes=0)
        #
        # # Fourier frequencies
        # omega = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
        # omega = omega[:, np.newaxis]
        omega, x_hat = fft(x, dt)

        # if self.spectral_cutoff:
        #     # Create mask
        #     mask = np.zeros([n, dims])
        #     for idx, cutoff in enumerate(self.spectral_cutoff):
        #         cutoff = n / 2 - cutoff * n
        #         cutoff = int(cutoff)
        #         mask[cutoff: -cutoff, idx] = 1  # Keep coeffs between cutoff indices
            # Truncate Fourier coefficients
            # x_hat = x_hat * mask

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

