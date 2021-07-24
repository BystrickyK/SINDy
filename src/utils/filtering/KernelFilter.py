import scipy.signal
from containers.DynaFrame import create_df
import numpy as np

class KernelFilter:
    def __init__(self, kernel='hann', kernel_size=8):
        self.kernel = kernel
        self.kernel_size = kernel_size

    # Convolution filtering
    def filter(self, x, var_label='x'):
        if self.kernel == 'hann':
            krnl = scipy.signal.hann(self.kernel_size)
        elif self.kernel == 'flattop':
            krnl = scipy.signal.flattop(self.kernel_size)
        else:
            return

        krnl /= sum(krnl)  # Normalize kernel

        x_filtered = np.apply_along_axis(
            lambda col: scipy.signal.convolve(col, krnl, mode='same'),
            0, x)
        return create_df(x_filtered, var_label=var_label)
