import numpy as np

def fft(x, dt):
    n = x.shape[0]  # Number of samples
    fs = 1/dt  # Sampling freq

    # Fourier coefficients
    x_hat = np.fft.fft(x, axis=0)
    x_hat = np.fft.fftshift(x_hat, axes=0)

    # Fourier frequencies in rad/s
    omega = (fs*2*np.pi/n) * np.arange(-n/2, n/2)
    omega = omega[:, np.newaxis]

    return omega, x_hat
