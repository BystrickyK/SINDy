import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'

def white_noise(N, dispersion, mean=0, distribution='gaussian', seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set the seed for identical results on every run

    if distribution == 'gaussian':
        # Dispersion is assumed to be a standard deviation
        sig = mean + dispersion * np.random.randn(N)
    elif distribution == 'uniform':
        # Dispersion is assumed to be range/2
        sig = mean - dispersion / 2 + dispersion * np.random.rand(N)
    return sig


def random_walk(N, dispersion, mean=0, freq_cutoff=1, filter_order=4, fs=400, distribution='gaussian', seed=None):
    # Generate white noise
    sig_white_noise = white_noise(N, dispersion, mean=mean, distribution=distribution, seed=seed)

    # Create a Butterworth filter
    filter_sos = signal.butter(filter_order, freq_cutoff, fs=fs, output='sos')

    # Filter the white noise signal
    sig_rand_walk = signal.sosfilt(filter_sos, sig_white_noise)

    return sig_rand_walk


class CallableSignal():
    """
    Acts as an interface to a digital signal, except takes time on the input (instead of the index) and interpolates
    if the time isn't exactly represented in the data
    """

    def __init__(self, sig, fs=400):
        self.sig = sig
        self.fs = fs
        self.t_step = 1 / fs

        self.eps = 10 * np.nextafter(0., 1.)  # very smol number

    def __call__(self, t):
        k_exact = t / self.t_step
        k_low = int(np.floor(k_exact))
        k_high = int(np.ceil(k_exact))

        if (k_high - k_exact) < self.eps:
            return self.sig[k_high]

        elif (k_exact - k_low) < self.eps:
            return self.sig[k_low]

        else:
            k = (k_low, k_high)
            yk = (self.sig[k_low], self.sig[k_high])

            y_interp = np.interp(k_exact, k, yk)
            return y_interp


class RandomWalk(CallableSignal):
    def __init__(self, t, dispersion=1, mean=0, freq_cutoff=1, filter_order=4, fs=400, distribution='gaussian', seed=None):
        """
        Creates a random walk signal and creates a function that takes time as input and returns the according signal value.
        Args:
            t: Signal length in seconds
            dispersion:
            freq_cutoff:
            filter_order:
            fs: Sampling frequency ( fs = 1/sampling_period )
            distribution:
            seed:
        """
        self.dispersion = dispersion
        self.freq_cutoff = freq_cutoff
        self.filter_order = filter_order
        self.distribution = distribution
        self.seed = seed
        self.mean = mean

        self.N = t*fs + 1  # Number of samples
        self.t = t

        sig = random_walk(self.N, dispersion, mean, freq_cutoff, filter_order, fs, distribution, seed=seed)
        CallableSignal.__init__(self, sig, fs)

    def plot(self):
        x = np.arange(0, self.t + self.eps, self.t_step)
        y = [self(xk) for xk in x]

        f, Pxx = signal.periodogram(y, self.fs)

        with plt.style.context({'./images/BystrickyK.mplstyle', 'seaborn'}):
            fig, axs = plt.subplots(2, 1, tight_layout=True)

            axs[0].plot(x, y, linewidth=2)
            axs[0].set_xlabel("Time [s]")

            axs[1].loglog(f[1:], Pxx[1:])  # The first element for f==0 -inf, so avoid it
            axs[1].set_xlabel('Frequency [Hz]')
            axs[1].set_ylabel('PSD')
            axs[1].vlines(self.freq_cutoff, ymin=min(Pxx[1:]), ymax=max(Pxx[1:]), linestyles='dashed',
                          color='black', linewidth=2)


# %%

# walk = RandomWalk(1000, 10, seed=0)
# t = np.arange(0, 1000 / 400, 1 / 400)
# t = t+0.00001
# t = t[1:-1]
# y = [walk(ti) for ti in t]
# plt.plot(t, y)

