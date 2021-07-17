import numpy as np

def ifft(x_hat):
    x_hat = np.fft.ifftshift(x_hat, axes=0)
    x = np.real(np.fft.ifft(x_hat, axis=0))
    return x
