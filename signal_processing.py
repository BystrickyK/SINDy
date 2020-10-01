import numpy as np
import scipy.signal
import pandas as pd


class Signal():
    def __init__(self, sim_data, noise_power=0):
        self.t = sim_data[:, 0]

        # DF of the original signal
        x_clean = sim_data[:, 1:]
        self.x_clean = self.state_df(x_clean)

        # DF of the original signal with added white noise
        x = x_clean + noise_power * np.random.randn(*x_clean.shape)
        self.x = self.state_df(x)

        # Number of samples (readings)
        self.samples = self.x.shape[0]
        # Number of dimensions
        self.dims = self.x.shape[1]
        # Sampling period
        self.dt = self.t[1]-self.t[0]

    def state_df(self, x):
        state_str = ['X'+str(i+1) for i in range(x.shape[1])]
        return pd.DataFrame(
            data=x,
            index=self.t,
            columns=state_str
        )

class ProcessedSignal(Signal):
    def __init__(self, sim_data,
                 spectral_cutoff = None,
                 kernel = None,
                 kernel_size = 8,
                 noise_power = 0,
                 model = None):

        Signal.__init__(self, sim_data, noise_power)

        # Various metadata
        self.info = {}

        # How many frequencies should be kept from each side of the spectrum (centered at 0 freq)
        # Ex: For 100, keeps 2*100+1 frequencies, from which 100 is positive, 100 negative, 1 zero
        if isinstance(spectral_cutoff, (int)):
            spectral_cutoff = [spectral_cutoff for dim in range(self.dims)]
        self.spectral_cutoff = spectral_cutoff


        # Spectral derivative
        self.dxdt_spectral = None
        self.spectral_derivative()  # Fills dxdt_spectral

        # Finite difference derivative
        self.dxdt_finitediff = None
        self.finite_difference_derivative()  # Fills dxdt_finitediff

        # Kernel smoothing
        self.kernel = kernel  # Which window should be used (from scipy.signals.windows)
        self.kernel_size = kernel_size  # Size of the window

        # Filtered spectral derivative
        self.dxdt_spectral_filtered = None
        self.spectral_derivative_filtered()

        # Calculate exact derivative from the system model (if available)
        self.dxdt_exact = None
        self.model = model
        self.exact_derivative()

    # Calculates the spectral derivative from self.x
    def spectral_derivative(self):
        L = self.t[-1]  #  Domain length (~ total time)
        n = self.samples

        # Fourier coefficients
        x_hat = np.fft.fft(self.x, axis=0)
        x_hat = np.fft.fftshift(x_hat, axes=0)
        self.info['fourier_coeffs'] = x_hat

        # Fourier frequencies
        omega = (2*np.pi/L) * np.arange(-n/2, n/2)
        omega = omega[:, np.newaxis]
        self.info['fourier_freqs'] = omega

        if self.spectral_cutoff:
            # Create mask
            mask = np.zeros([self.samples, self.dims])
            for idx, cutoff in enumerate(self.spectral_cutoff):
                cutoff = self.samples/2-cutoff
                cutoff = int(cutoff)
                mask[cutoff: -cutoff, idx] = 1
            self.info['fourier_coeffs'] = x_hat
            # Truncate Fourier coefficients
            x_hat = x_hat * mask
            self.info['fourier_coeffs_truncated'] = x_hat

        dxdt_hat = 1j*omega*x_hat
        dxdt_hat = np.fft.ifftshift(dxdt_hat, axes=0)
        dxdt = np.real(np.fft.ifft(dxdt_hat, axis=0))
        self.dxdt_spectral = self.state_df(dxdt)

    # Calculates the finite difference derivative
    def finite_difference_derivative(self, direction='forward'):
        if direction == 'forward':
            dxdt = (np.diff(self.x, axis=0))/self.dt  # last value is missing
            dxdt = np.vstack((dxdt, dxdt[-1,:]))
            self.dxdt_finitediff = self.state_df(dxdt)
        elif direction == 'backward':
            x = np.flip(self.x.values, axis=0)
            dxdt = (-np.diff(x, axis=0))/self.dt
            dxdt = np.flip(dxdt, axis=0)  # first value is missing
            dxdt = np.vstack((dxdt[0,:], dxdt))
            self.dxdt_finitediff = self.state_df(dxdt)

    def spectral_derivative_filtered(self):
        if self.kernel == 'hann':
            krnl = scipy.signal.hann(self.kernel_size)
        elif self.kernel == 'flattop':
            krnl = scipy.signal.flattop(self.kernel_size)
        else:
            return

        krnl /= sum(krnl)  # Normalize kernel

        dxdt_spectral_filtered = np.apply_along_axis(
            lambda col: scipy.signal.convolve(col, krnl, mode='same'),
            0,
            self.dxdt_spectral)
        self.dxdt_spectral_filtered = self.state_df(dxdt_spectral_filtered)

    def exact_derivative(self):
        dxdt = np.array([*map(self.model, self.x_clean.values)])
        self.dxdt_exact = self.state_df(dxdt)