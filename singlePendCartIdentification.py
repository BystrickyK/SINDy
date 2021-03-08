import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import plot_corr
from itertools import combinations
import matplotlib

matplotlib.use('Qt5Agg')

filename = './singlePendulumCart/singlePend.csv'
sim_data = pd.read_csv(filename)

X = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1], noise_power=0)
dX = StateDerivativeSignal(X)
u = ForcingSignal(sim_data['t'], sim_data.iloc[:, -1])

state_data = X.x
state_derivative_data = dX.dx
input_data = u.u

dim = state_data.shape[1]

state_derivative_data = state_derivative_data.iloc[:, dim//2:]  # Remove ambiguity between x3,x4 and dx1,dx2
#%%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
trig_data = trigonometric_library(state_data.iloc[:, 1:dim//2])

# linear/angular accelerations -> second half of state var derivatives

#%%
trig_state_derivative = product_library(trig_data, state_derivative_data)

# Function library Theta
theta = pd.concat([state_data, state_derivative_data,
                   trig_data, trig_state_derivative], axis=1)

cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]

corr = theta.corr()
plot_corr(corr, theta.columns)
plt.show()

# x4 = state_data.iloc[cutoff:-cutoff,3]
# dx2 = state_derivative_data.iloc[cutoff:-cutoff,1]
# plt.plot(x4, color='blue')
# plt.plot(dx2, color='red')
# plt.legend(['x4','dx2'])
# plt.show()

EqnIdentifier = PI_Identifier(theta)
EqnIdentifier.set_thresh_range(lims=(0.1, 1), n=10)
EqnIdentifier.create_models(n_models=10, iters=8)