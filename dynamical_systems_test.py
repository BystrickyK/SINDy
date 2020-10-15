from dynamical_systems import LotkaVolterraSystem
import numpy as np
import matplotlib.pyplot as plt
from signal_processing import StateSignal, ForcingSignal

sys = LotkaVolterraSystem(x0=[3, 5])
sys.propagate(10)

u_f1 = lambda t, x: 2 * (1 - x[0])
u_f2 = lambda t, x: np.sin(t)**2  # rabbit feeding campaigns
u = (u_f1, u_f2)

sys.propagate_forced(30, u)

x = StateSignal(sys.sim_data[:, [0, 1, 2]])
u = ForcingSignal(sys.sim_data[:, [0, 3, 4]])
# fig, axs = plt.subplots(2, 1, tight_layout=True)
# axs[0].plot(sys.sim_data[:, 0], sys.sim_data[:, 1], 'r')
# axs[0].plot(sys.sim_data[:, 0], sys.sim_data[:, 2], 'b')
# axs[0].legend(['Predators','Prey'])
# axs[1].plot(sys.sim_data[:, 0], sys.sim_data[:, 3], 'r')
# axs[1].plot(sys.sim_data[:, 0], sys.sim_data[:, 4], 'b')
# plt.show()

