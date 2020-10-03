import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt


class Signal():
    def __init__(self, sim_data, noise_power=0):
        self.t = sim_data[:, 0]

        # DF of the original signal
        x_clean = sim_data[:, 1:]
        self.x_clean = self.state_df(x_clean)

        # The DataFrame self.x is calculated from self.x_clean via
        # the noise_power setter method
        self.x = None
        self.noise_power = noise_power

        # Number of samples (readings)
        self.samples = self.x.shape[0]
        # Number of dimensions
        self.dims = self.x.shape[1]
        # Sampling period
        self.dt = self.t[1]-self.t[0]

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        x = self.x_clean + noise_power * np.random.randn(*self.x_clean.shape)
        self.x = self.state_df(x)
        self._noise_power = noise_power

    def state_df(self, x):
        state_str = ['x'+str(i+1) for i in range(x.shape[1])]
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

        self.svd = {}
        self.compute_svd()

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


    def plot_dxdt_comparison(self):
        # Plot analytic and spectral derivatives
        t = self.t
        dxdt_exact = self.dxdt_exact.values
        dxdt_spectral = self.dxdt_spectral.values
        dxdt_spectral_filtered = self.dxdt_spectral_filtered.values
        dxdt_findiff = self.dxdt_finitediff.values
        with plt.style.context('seaborn-colorblind'):
            fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
            plt.xlabel('Time t [s]')
            ylabels = [r'$\.{X}_{' + str(i + 1) + '} [s^{-1}]$' for i in range(self.dims)]
            for ii, ax in enumerate(axs):
                ax.plot(t, dxdt_exact[:, ii], 'k', alpha=1, linewidth=2, label='Exact')
                ax.plot(t, dxdt_spectral[:, ii], '-', color='blue', alpha=0.8, linewidth=2, label='Spectral Cutoff')
                ax.plot(t, dxdt_findiff[:, ii], '-', color='c', alpha=0.5, label='Forward Finite Difference')
                ax.plot(t, dxdt_spectral_filtered[:, ii], '-', color='red', alpha=0.8, linewidth=2,
                        label='Spectral Cutoff Filtered')
                ax.set_ylabel(ylabels[ii])
                ax.legend(loc=1)

    def compute_svd(self):
        u,s,vt = np.linalg.svd(self.x.values.T, full_matrices=False)
        s = np.diag(s)
        self.svd['U'] = u
        self.svd['Sigma'] = s
        self.svd['V*'] = vt

    def plot_svd(self):
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(10, 6)

        p0 = axs[0].matshow(self.svd['U'], cmap='cividis')
        plt.colorbar(p0, ax=axs[0])
        axs[0].set_xticks([*range(0,self.dims)])
        axs[0].set_xticklabels(['PC'+str(i+1) for i in range(self.dims)])
        axs[0].set_yticks([*range(0,self.dims)])
        axs[0].set_yticklabels(['X'+str(i+1) for i in range(self.dims)])
        axs[0].set_title("Left Singular Vectors\nPrincipal Components")

        p1 = axs[1].matshow(self.svd['Sigma'], cmap='viridis')
        plt.colorbar(p1, ax=axs[1])
        axs[1].set_xticks([*range(0,self.dims)])
        axs[1].set_xticklabels([str(i+1) for i in range(self.dims)])
        axs[1].set_yticks([*range(0,self.dims)])
        axs[1].set_yticklabels([str(i+1) for i in range(self.dims)])
        axs[1].set_title("Singular Values")
