from dynamical_systems import LorenzSystem, LotkaVolterraSystem
import numpy as np
import matplotlib.pyplot as plt
from signal_processing import StateSignal, ForcingSignal
from function_library_creators import poly_library
import pandas as pd

sys = LotkaVolterraSystem(x0=[3, 5])
sys.propagate(10)

u_f1 = lambda t, x: 2 * (1 - x[0])
u_f2 = lambda t, x: np.sin(t)**2  # rabbit feeding campaigns
u = (u_f1, u_f2)

sys.propagate_forced(30, u)

x = StateSignal(sys.sim_data[:, [0, 1, 2]])
u = ForcingSignal(sys.sim_data[:, [0, 3, 4]])

lib = poly_library(x.x, poly_orders=(1, 2))

lib2 = pd.concat([lib, u.u], axis=1)
