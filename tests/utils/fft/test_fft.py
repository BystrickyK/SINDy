from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
fs = int(1/dt)
N = 2048
max_freq = N * fs
t = np.linspace(0, 1024, 2*fs, endpoint=False)
x = np.cos(1*2*np.pi*t) + np.cos(2*2*np.pi*t) + np.sin(8*2*np.pi*t)

omega, x_hat = fft(x, dt)
freq_idx = lambda value: (np.abs(omega - value)).argmin()
freq_neighborhood_idx = lambda freq: range(freq_idx(freq)-20, freq_idx(freq)+20)
freq_neighborhood_amp = lambda freq: np.sqrt(np.sum(np.abs(x_hat[freq_neighborhood_idx(freq)])))

psd_est = np.abs(x_hat)
plt.plot(omega, psd_est, 'o', color='tab:red')
plt.xlim([-20, omega[-1]])
plt.show()

plt.figure()
w = omega[int(len(omega)/2):-20:8]
plt.plot(w, [freq_neighborhood_amp(v) for v in w], 'o')
plt.show()

#%%

def test_ifft_of_fft_equality():
    x_expected = x
    [_, x_hat] = fft(x_expected, dt)
    x_real = ifft(x_hat)
    assert np.all(np.isclose(x_real, x_expected))

def test_fft_of_ifft_equality():
    x_hat_expected = np.zeros(256, dtype='complex128')
    x_hat_expected[0:15] = 2-3*1j
    x_hat_expected[15] = 1+1j
    x_hat_expected = np.concatenate([x_hat_expected[::-1], x_hat_expected.conjugate()])
    x = ifft(x_hat_expected)
    omega, x_hat_real = fft(x, 1)
    assert np.all(np.isclose(x_hat_real,x_hat_expected))

