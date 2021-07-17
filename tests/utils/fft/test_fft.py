from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
t = np.arange(0, 10000, dt)
x = np.sin(3*t) + 2*np.sin(6*t)

omega, x_hat = fft(x, dt)
freq_idx = lambda value: (np.abs(omega - value)).argmin()
freq_neighborhood_idx = lambda freq: range(freq_idx(freq)-30, freq_idx(freq)+30)
freq_neighborhood_amp = lambda freq: np.sqrt(np.sum(np.abs(x_hat[freq_neighborhood_idx(freq)])))

psd_est = np.abs(x_hat)
plt.plot(omega, psd_est, 'o')
plt.xlim([0, 9])

plt.figure()
w = np.arange(0, 15, 0.25)
plt.plot(w, [freq_neighborhood_amp(v) for v in w], 'o')

#%%

def test_ifft_of_fft_equality():
    x_expected = x
    [_, x_hat] = fft(x_expected, dt)
    x_real = ifft(x_hat)
    assert np.all(np.isclose(x_real, x_expected))

def test_fft_of_ifft_equality():
    x_hat_expected = np.zeros(256, dtype='complex128')
    x_hat_expected[15] = 1+1j
    x_hat_expected = np.concatenate([x_hat_expected[::-1], x_hat_expected.conjugate()])
    x = ifft(x_hat_expected)
    omega, x_hat_real = fft(x, 1)

    assert np.isclose( np.sum(np.abs(x_hat_real)), np.sum(np.abs(x_hat_expected)))

