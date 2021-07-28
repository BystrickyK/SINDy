from src.utils.filtering.SpectralFilter import SpectralFilter
from filtering.KernelFilter import KernelFilter
from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from definitions import ROOT_DIR
import copy
import os


stylepath = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
plt.style.use({'seaborn', stylepath})
np.random.seed(0)

def random_complex(size):
    # To assure symmetry, generate only half of the signal and define the other half as the mirror image
    x_hat_complex = np.random.rand(int(size/2), 2).astype('complex64') - 0.5
    x_hat_complex[:, 1] *= 1j  # multiply second column by imag unit
    x_hat_complex = np.sum(x_hat_complex, axis=1)  # sum real and imag part
    x_hat_complex /= np.abs(x_hat_complex)  # normalize
    x_hat_complex = np.concatenate([np.conjugate(x_hat_complex)[::-1],
                                    np.array([1+0*1j]),
                                    x_hat_complex[:-1]]) # append mirror image
    # fig, axs = plt.subplots(nrows=2, tight_layout=True, sharex=True)
    # axs[0].plot(np.real(x_hat_complex))
    # axs[1].plot(np.imag(x_hat_complex))
    # plt.xticks(range(len(x_hat_complex)))
    # plt.xlim([len(x_hat_complex)/2-10, len(x_hat_complex)/2+10])
    # plt.show()
    # plt.polar(np.angle(x_hat_complex), np.abs(x_hat_complex), '.')
    # plt.show()
    return x_hat_complex

def create_fourier_signal(e1=10, e2=60, max_freq=256, dt=0.001):
    # inputs -> indexes of the edges of the fourier image

    np.random.seed(0)
    # the shape of the fourier image signal
    tmp = np.arange(-e2, e2, 1)
    signal = np.zeros_like(tmp) + 1e-6
    mid = int(len(tmp)/2)
    signal[mid: mid+e1] = mid
    signal[mid+e1:mid+e2] = mid - tmp[mid+e1:mid+e2] + tmp[mid+e1]
    signal[0:mid] = signal[:mid-1:-1]  # mirror the above signal
    signal -= np.min(signal)
    signal = signal / np.max(signal)


    # frequencies
    omega = 1 / dt * 2 * np.pi * np.arange(-max_freq, max_freq) / (2*max_freq)
    zfi = np.where(omega == 0)[0][0]  # zero frequency index

    # Random complex numbers (for phase information)
    x_hat_complex = random_complex(len(omega))

    # signal in the frequency domain
    x_hat = np.zeros_like(omega, dtype='complex64') + 1e-6
    x_hat[zfi-e2:zfi+e2] = signal * np.sqrt(10)
    x_hat = x_hat.astype('complex64') * x_hat_complex

    return x_hat, omega

def plot_psd(xn_hat, x_hat, n_hat, omega):
    noise_mean_amp = np.abs(n_hat)
    psd_summed = np.abs(xn_hat)**1
    psd_sig = np.abs(x_hat)**1
    psd_noise = np.abs(n_hat)**1
    lower_bound = np.abs((np.abs(x_hat) - np.abs(noise_mean_amp))) ** 1
    upper_bound = np.abs((np.abs(x_hat) + np.abs(noise_mean_amp))) ** 1

    fig, axs = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(10,6))
    axs.plot(omega, lower_bound, '-.', color='tab:grey', label='signal+noise, fully destructive',
             linewidth=1.5, alpha=0.75)
    axs.plot(omega, upper_bound, '--', color='tab:grey', label='signal+noise, fully constructive',
             linewidth=1.5, alpha=0.75)
    axs.plot(omega, psd_summed, color='black', alpha=0.5, linewidth=2, label='signal+noise')
    axs.plot(omega, psd_sig, color='tab:blue', alpha=0.5, label='signal')
    axs.plot(omega, psd_noise, color='tab:red', alpha=0.5, label='noise')
    axs.legend()
    axs.set_xlabel("Frequency $\omega\ [\\frac{rad}{s}]$")
    axs.set_ylabel("Amplitude $A_\omega$")
    axs.set_yscale('linear')
    # axs[1].plot(omega, snr, linewidth=3, color='black', label='SNR')
    # axs[1].legend()
    plt.show()

def plot_time_signal(x, n, dt=0.001):
    t = np.arange(0, len(x), 1) * dt

    fig, axs = plt.subplots(tight_layout=True, figsize=(10,6))
    axs.plot(t, x+n, label='signal+noise', linewidth=2, alpha=0.5, color='black')
    axs.plot(t, x, label='signal', color='tab:blue')
    axs.plot(t, n, label='noise', color='tab:red')
    axs.legend()
    axs.set_xlabel("Time $t\ [s]$")
    axs.set_ylabel("Value")
    plt.xlim([151*dt, 301*dt])
    plt.show()

def plot_comparison(x_clean, x_filtered, x_noisy, dt=0.001):
    t = np.arange(0, len(x_clean), 1) * dt

    x_clean = np.array(x_clean)
    x_filtered = np.array(x_filtered)
    x_noisy = np.array(x_noisy)

    fig, axs = plt.subplots(tight_layout=True, figsize=(10,6))
    axs.plot(t, x_clean, label='Clean signal', linewidth=2, alpha=0.5, color='tab:blue')
    axs.plot(t, x_filtered, label='Filtered signal', linewidth=2, alpha=0.75, color='tab:red')
    axs.plot(t, x_noisy, label='Measured signal', linewidth=2, alpha=0.5, color='tab:grey')
    axs.legend()
    axs.set_xlabel("Time $t\ [s]$")
    axs.set_ylabel("Value")
    plt.xlim([151*dt, 301*dt])
    plt.show()

def plot_filt_comparison(x_clean, x_hann5, x_hann9, x_spec, x_noisy, dt=0.001):
    t = np.arange(0, len(x_clean), 1) * dt

    x_clean = np.array(x_clean).reshape(-1)
    x_hann5 = np.array(x_hann5).reshape(-1)
    x_hann9 = np.array(x_hann9).reshape(-1)
    x_spec = np.array(x_spec).reshape(-1)
    x_noisy = np.array(x_noisy).reshape(-1)

    fig, axs = plt.subplots(nrows=2, sharex=True, tight_layout=True, figsize=(10,12))
    axs[0].plot(t, x_clean, label='Clean signal', linewidth=2, alpha=0.5, color='tab:blue')
    axs[0].plot(t, x_noisy, label='Measured signal', linewidth=2, alpha=0.5, color='tab:grey')
    axs[0].plot(t, x_hann5, '--', label='Hann 5', linewidth=2, alpha=0.75, color='tab:red')
    axs[0].plot(t, x_hann9, ':', label='Hann 9', linewidth=2, alpha=0.75, color='tab:red')
    axs[0].plot(t, x_spec, label='Spectral', linewidth=2, alpha=0.75, color='tab:purple')
    axs[0].legend()
    axs[-1].set_xlabel("Time $t\ [s]$")
    axs[0].set_ylabel("Value")
    plt.xlim([201*dt, 301*dt])
    diff = lambda sig: np.abs(sig-x_clean)
    axs[1].plot(t, diff(x_noisy), label='Measured signal',
                        linewidth=2, alpha=0.5, color='tab:grey')
    axs[1].plot(t, diff(x_hann5), '--', label='Hann 5', linewidth=2, alpha=0.75, color='tab:red')
    axs[1].plot(t, diff(x_hann9), ':', label='Hann 9', linewidth=2, alpha=0.75, color='tab:red')
    axs[1].plot(t, diff(x_spec), label='Spectral', linewidth=2, alpha=0.75, color='tab:purple')
    axs[1].set_ylabel("Absolute error")
    plt.show()

def plot_periodogram(x_hat, xn_hat, xf_hat, omega):
    x_psd = np.abs(x_hat)**2
    xn_psd = np.abs(xn_hat)**2
    xf_psd = np.abs(xf_hat)**2

    fig, axs = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(10,6))
    axs.plot(omega, x_psd, label='signal', color='tab:blue')
    axs.plot(omega, xn_psd, label='signal+noise', color='tab:grey')
    axs.plot(omega, xf_psd, label='signal+noise after filtering', linestyle=':',
             color='tab:red')
    axs.legend()
    axs.set_xlabel("Frequency $\omega\ [\\frac{rad}{s}]$")
    axs.set_ylabel("Power $A_\omega$")
    plt.show()

#%%
# Create signal with limited bandwidth
max_freq = 512
x_hat_clean, omega = create_fourier_signal(64, 90, max_freq)

dt = 0.001

# Add white noise
n_amp = 0.25  # noise amplitude
n_hat = random_complex(len(omega)) * n_amp # white noise in fourier domain

# combined signal
x_hat_noisy = x_hat_clean + n_hat

plot_psd(x_hat_noisy, x_hat_clean, n_hat, omega)

x_clean = np.real(ifft(x_hat_clean))
x_noisy = np.real(ifft(x_hat_noisy))
n = ifft(n_hat)

plot_time_signal(x_clean, n)
# #%%
# s = 2**4
# np.random.seed(0)
# a = np.random.rand(s) - 0.5
# _, b = fft(a, 1)
# c = random_complex(s)
# f, ax = plt.subplots(nrows=4)
# ax[0].plot(np.real(b), color='tab:green')
# ax[0].plot(np.real(c), color='tab:red')
# ax[1].plot(np.imag(b), color='tab:green')
# ax[1].plot(np.imag(c), color='tab:red')
# ax[2].plot(np.angle(b), color='tab:green')
# ax[2].plot(np.angle(c), color='tab:red')
# ax[3].plot(np.abs(b), color='tab:green')
# ax[3].plot(np.abs(c), color='tab:red')
# [ax[i].vlines((s-1)/2, -3, 3) for i in (0,1,2,3)]

# plt.figure()
# tmp1 = b[s//2:]
# tmp2 = -b[:s//2][::-1]
# plt.plot(tmp1, '-')
# plt.plot(tmp2, '_-')
# tmp = ifft(b)

#%%
x_noisy_df = pd.DataFrame(x_noisy)
filter = SpectralFilter(x_noisy_df, dt, plot=True)
filter.find_cutoff_frequencies()
x_filtered = filter.filter()
w, x_hat_filtered = fft(x_filtered, dt)

plot_periodogram(x_hat_clean, x_hat_noisy, x_hat_filtered, omega)
plot_comparison(x_clean, x_filtered, x_noisy)

#%%
def hann(x, size):
    filter_conv = KernelFilter(kernel='hann', kernel_size=size)
    xf = filter_conv.filter(x)
    _, xf_hat = fft(xf, 1)
    return xf, xf_hat

x_hann5, x_hat_hann5 = hann(x_noisy, 5)
x_hann9, x_hat_hann9 = hann(x_noisy, 9)

plot_periodogram(x_hat_clean, x_hat_noisy, x_hat_hann9, omega)
plot_comparison(x_clean, x_hann9, x_noisy)
# plot_periodogram(x_hat_clean, x_hat_noisy, x_hat_filtered_kern, omega)
#%%
plot_filt_comparison(x_clean, x_hann5, x_hann9, x_filtered, x_noisy)
#%%

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



