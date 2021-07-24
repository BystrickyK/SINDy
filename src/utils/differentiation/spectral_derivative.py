from src.utils.fft.fft import fft
from src.utils.fft.ifft import ifft
from tools import mirror, halve
import numpy as np
import pandas as pd

# Calculates the spectral derivative from x
def compute_spectral_derivative(x, dt, mirroring=True):
    """
    x (DataFrame): State measurements
    dt (Float): Sampling period
    """

    # if isinstance(x, pd.DataFrame):
    x = np.array(x)

    if mirroring:
        x = mirror(x)

    omega, x_hat = fft(x, dt)

    dxdt_hat = 1j * omega * x_hat  # Fourier image of the derivative of x
    dxdt = np.real(ifft(dxdt_hat))

    if mirroring:
        dxdt = halve(dxdt)

    return dxdt
