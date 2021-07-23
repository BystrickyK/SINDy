from src.utils.filtering.SpectralFilter import SpectralFilter
from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os

plt.style.use({'seaborn', '../../../src/utils/visualization/BystrickyK.mplstyle'})
np.random.seed(0)

def random_complex(size):
    x_hat_complex = np.random.rand(size, 2).astype('complex64') - 0.5
    x_hat_complex[:, 1] *= 1j  # multiply second column by imag unit
    x_hat_complex = np.sum(x_hat_complex, axis=1)  # sum real and imag part
    x_hat_complex /= np.abs(x_hat_complex)  # normalize
    return x_hat_complex

def create_fourier_signal(e1=10, e2=60, max_omega=151):
    # inputs -> indexes of the edges of the fourier image

    np.random.seed(0)
    # the shape of the fourier image signal
    tmp = np.arange(-e2, e2, 1)
    signal = np.zeros_like(tmp) + 1e-9
    mid = int(len(tmp)/2)
    signal[mid: mid+e1] = mid
    signal[mid+e1:mid+e2] = mid - tmp[mid+e1:mid+e2] + tmp[mid+e1]
    signal[0:mid] = signal[:mid-1:-1]  # mirror the above signal
    signal -= np.min(signal)
    signal = signal / np.max(signal)


    # frequencies
    omega = np.arange(-max_omega, max_omega, 1)  # rad/s
    zfi = np.where(omega == 0)[0][0]  # zero frequency index

    # Random complex numbers (for phase information)
    x_hat_complex = random_complex(len(omega))

    # signal in the frequency domain
    x_hat = np.zeros_like(omega, dtype='complex64')
    x_hat[zfi-e2:zfi+e2] = signal * np.sqrt(10)
    x_hat = x_hat.astype('complex64') * x_hat_complex

    return x_hat, omega

def plot_psd(xn_hat, x_hat, n_hat):
    noise_mean_amp = np.mean(np.abs(n_hat))
    psd_summed = np.abs(xn_hat)**2
    psd_sig = np.abs(x_hat)**2
    psd_noise = np.abs(n_hat)**2
    lower_bound = (np.abs(x_hat) - np.abs(noise_mean_amp)) ** 2
    upper_bound = (np.abs(x_hat) + np.abs(noise_mean_amp)) ** 2

    snr = np.array(psd_sig) / np.array(psd_noise)

    fig, axs = plt.subplots(nrows=2, ncols=1, tight_layout=True)
    axs[0].plot(omega, lower_bound, '-.', color='tab:red', label='signal+noise, fully destructive')
    axs[0].plot(omega, upper_bound, '--', color='tab:red', label='signal+noise, fully constructive')
    axs[0].plot(omega, psd_summed, color='tab:red', alpha=0.5, linewidth=1.5, label='signal+noise')
    axs[0].plot(omega, psd_sig, color='tab:blue', label='signal')
    axs[0].plot(omega, psd_noise, color='tab:green', label='noise')
    axs[0].legend()
    axs[0].set_xlabel("Frequency $\omega$")
    axs[0].set_ylabel("Power $p_x$")
    axs[1].plot(omega, snr, linewidth=3, color='black', label='SNR')
    axs[1].legend()
    plt.show()

def plot_time_signal(x, n, t=None):
    if t is None:
        t = np.arange(0, len(x), 1)

    fig, axs = plt.subplots(tight_layout=True)
    axs.plot(t, x+n, label='signal+noise')
    axs.plot(t, x, label='signal')
    axs.plot(t, n, label='noise')
    axs.legend()
    axs.set_xlabel("Time $t$")
    axs.set_ylabel("Value")
    plt.show()

def plot_periodogram(x_hat, xn_hat, xf_hat, omega, dt=None):
    if dt is None:
        dt = 1

    x_psd = np.abs(x_hat)**2
    xn_psd = np.abs(xn_hat)**2
    xf_psd = np.abs(xf_hat)**2

    fig, axs = plt.subplots(nrows=1, ncols=1, tight_layout=True)
    axs.plot(omega, x_psd, label='signal')
    axs.plot(omega, xn_psd, label='signal+noise')
    axs.plot(omega, xf_psd, label='signal+noise after filtering', linestyle=':')
    axs.legend()
    axs.set_xlabel("Frequency $\omega$")
    axs.set_ylabel("Power $p_x$")
    axs.set_title("Periodogram")
    plt.show()

#%%
# Create signal with limited bandwidth
omega_max = 1024
x_hat_clean, omega = create_fourier_signal(64, 90, omega_max)

dt = 0.001

# Add white noise
n_amp = 0.05  # noise amplitude
n_hat = random_complex(len(omega)) * n_amp  # white noise in fourier domain

# combined signal
x_hat_noisy = x_hat_clean + n_hat

plot_psd(x_hat_noisy, x_hat_clean, n_hat)

x_clean = ifft(x_hat_clean)
x_noisy = ifft(x_hat_noisy)
n = ifft(n_hat)

plot_time_signal(x_clean, n)

#%%
x_noisy_df = pd.DataFrame(x_noisy)
filter = SpectralFilter(x_noisy_df, dt, plot=True)
filter.find_cutoff_frequencies()
x_filtered = filter.filter()
w, x_hat_filtered = fft(x_filtered, 1)

#%%
plot_periodogram(x_hat_clean, x_hat_noisy, x_hat_filtered, omega)

#%%
fig, axs = plt.subplots(tight_layout=True)
t = np.arange(0, len(x_clean), 1) * dt
axs.plot(t, x_clean, label='signal')
axs.plot(t, x_noisy, label='signal+noise')
axs.plot(t, x_filtered, '--', label='signal denoised')
axs.legend()
axs.set_xlabel("Time $t$")
axs.set_ylabel("Value")
plt.show()
print('end')

def test_signal_filtering():
    filter = SpectralFilter(x_noisy_df, dt, plot=False)
    filter.find_cutoff_frequencies()
    x_filtered = filter.filter()
    x_expected = x_filtered.reshape((-1,))
    x_real = x_clean
    tolerance = np.abs(np.max(x_real)) * 0.025
    isclose = np.isclose(x_real, x_expected, atol=tolerance, rtol=0)

    fig, ax = plt.subplots(nrows=2, tight_layout=True, sharex=True)
    ax[0].plot(x_expected, 'g')
    ax[0].plot(x_real, 'b')
    ax[0].plot(x_real+tolerance, 'r--')
    ax[0].plot(x_real-tolerance, 'r--')
    ax[1].plot(isclose)

    assert np.all(isclose)



