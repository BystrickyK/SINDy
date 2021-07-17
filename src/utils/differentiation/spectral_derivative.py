from src.utils.fft import fft, ifft
import numpy as np
import pandas as pd

# Calculates the spectral derivative from x
def compute_spectral_derivative(x, dt):
    """
    x (DataFrame): State measurements
    dt (Float): Sampling period
    """

    if isinstance(x, pd.DataFrame):
        x = np.array(x)

    omega, x_hat = fft(x, dt)

    dxdt_hat = 1j * omega * x_hat  # Fourier image of the derivative of x
    dxdt = ifft(dxdt_hat)

    return dxdt
