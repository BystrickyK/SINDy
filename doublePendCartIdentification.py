import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import plot_corr
from itertools import combinations
from scipy.spatial import distance_matrix
import matplotlib

matplotlib.use('Qt5Agg')

filename = './doublePendulumCart/doublePendSimData.csv'
sim_data = pd.read_csv(filename)

X = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1], noise_power=0)
dX = StateDerivativeSignal(X)
u = ForcingSignal(sim_data['t'], sim_data.iloc[:, -1])

state_data = X.x
state_derivative_data = dX.dx
input_data = u.u

dim = state_data.shape[1]

# %%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
angle_sums = sum_library(state_data.iloc[:, 1:dim // 2], (-2, -1, 0, 1, 2))
trig_data = trigonometric_library(angle_sums)
trig_data, rmvd = remove_twins(trig_data)

# linear/angular velocities -> second half of state vars
velocity_data = state_data.iloc[:, -dim // 2:]
vel_sq_data = square_library(velocity_data)

# linear/angular accelerations -> second half of state var derivatives
acceleration_data = state_derivative_data.iloc[:, -dim // 2:]

# %%
trig_vel = product_library(trig_data, velocity_data)
trig_vel_sq = product_library(trig_data, vel_sq_data)
trig_accel = product_library(trig_data, acceleration_data)

# Function library Theta
theta = pd.concat([state_data, acceleration_data,
                   trig_data, trig_vel, trig_vel_sq, trig_accel], axis=1)


cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]

corrmat = theta.corr()
plt.imshow(corrmat)
# plot_corr(corrmat, theta.columns)

EqnIdentifier = PI_Identifier(theta)
EqnIdentifier.set_thresh_range(lims=(0.01, 3), n=15)
EqnIdentifier.create_models(n_models=89, iters=5)

# %%
models = EqnIdentifier.all_models
sols = []
for model in models:
    # 0 -> lhs string; 1 -> rhs string; 2 -> rhs solution; 3 -> complexity
    lhs_guess_idx = list(theta.columns).index(model[0])
    full_sol = list(model[2])
    full_sol.insert(lhs_guess_idx, -1)
    active_regressors = np.array(full_sol)!=0
    sols.append(active_regressors)

for model in models:
    model[2] = model[2].round(4)
sol_print = lambda sol: print(f'{" + ".join([str(param) + term for param, term in np.array([*zip(model[2], model[1])])[model[2]!=0]])} = 0')

sols = np.array(sols)
corr = np.corrcoef(sols)
plt.imshow(corr)