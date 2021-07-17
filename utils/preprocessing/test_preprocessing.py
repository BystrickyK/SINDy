from utils.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 100, 0.01)
sig1 = np.sin(t)**2 + t**0.7
sig2 = np.cos(t) + np.sin(t**2)

fig, axs = plt.subplots(nrows=2, ncols=1, tight_layout=True)
axs[0].plot(t, sig1)
axs[1].plot(t, sig2)
