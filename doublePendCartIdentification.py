import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import *
from utils.solution_processing import *
import matplotlib
import pickle
import os
from decimal import Decimal


matplotlib.use('Qt5Agg')

dirname = '.' + os.sep + 'doublePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata2.csv'

sim_data = pd.read_csv(filename)
dt = sim_data.iloc[1, 0] - sim_data.iloc[0, 0]

Filter = KernelFilter(kernel='hann', kernel_size=31)
X = StateSignal(sim_data.iloc[:, 1:-1], dt=dt, relative_noise_power=0.01)
# X = Filter.filter(X.values, 'x')  # Noisy data after filtering
X = X.values_clean  # Use clean data

X = StateSignal(X, dt)
DX = StateDerivativeSignal(X, method='spectral')
u = ForcingSignal(sim_data.iloc[:, -1], dt)

# state_data = X.x
state_data = X.values
state_derivative_data = DX.values
input_data = u.values

# Resample, take only every n-th element
n = 3
state_data = state_data.iloc[::n, :].reset_index().drop('index', axis=1)
state_derivative_data = state_derivative_data.iloc[::n, :].reset_index().drop('index', axis=1)
input_data = input_data.iloc[::n].reset_index().drop('index', axis=1)

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
theta = pd.concat([velocity_data, acceleration_data,
                   trig_data, trig_vel, trig_vel_sq, trig_accel], axis=1)


cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]
theta = theta.astype('float32')

# cols = ['sin(2*x_2 + -1*x_3)', 'sin(1*x_3)',
#         'x_5', 'x_6', 'dx_6', 'cos(2*x_2 + -2*x_3)*dx_6',
#         'sin(1*x_2 + -1*x_3)*x_5*x_5', 'cos(2*x_2 + -1*x_3)*dx_4',
#         'cos(1*x_3)*dx_4', 'sin(2*x_2 + -2*x_3)*x_6*x_6',
#         'cos(1*x_2 + -1*x_3)*x_5', 'cos(1*x_2 + -1*x_3)*x_6']
# theta = theta.loc[:, cols]

# Identify equation for dx_6
idx = np.array([('dx_5' in col) for col in theta.columns]) # Find equations with dx_5
theta = theta.loc[:, ~idx]

corrmat = theta.corr()
plot_corr(corrmat, theta.columns, labels=False)

cachename = dirname + 'doublePendSolutions_dX6_EnergyThreshErrorWeighted1x'
rewrite = False
if os.path.exists(cachename) and not rewrite:
    print("Retrieving solution from cache.")
    with open(cachename, 'rb') as f:
        models = pickle.load(f)
else:
    print("No solution in cache, calculating solution from scratch.")
    EqnIdentifier = PI_Identifier(theta)
    EqnIdentifier.set_thresh_range(lims=(0.00001, 0.3), n=5)
    EqnIdentifier.create_models(n_models=theta.shape[1], iters=10, shuffle=False)
    models = EqnIdentifier.all_models
    with open(cachename, 'wb') as f:
        pickle.dump(models, f)


#%% Remove duplicate models
models = unique_models(models, theta.columns)

#%% Visualize the solutions -> calculate and plot activation distance matrix
# and plot the matrix of implicit solutions
dist = distance_matrix(models, plot=False)
# plot_implicit_sols(models, theta.columns)
#%% Look for consistent models by finding clusters in the term activation space
models = consistent_models(models, dist,
                           min_cluster_size=2)

# gsols = np.vstack(models.sol.values)
# theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
# gsols = gsols[:, theta_terms_idx]
# glhs_guess_str = models.lhs.values

plot_implicit_sols(models, theta.columns, show_labels=False)

#%% Decompose one of the models
choice = 3
model = models.iloc[choice, :]

active_terms = theta.iloc[:, model['active']].values
term_labels = theta.columns[model['active']]
parameters = np.array(model['sol'])[model['active']]
signals = parameters*active_terms
solution_string = ' + \n'.join(['$' + str(round(par,3)) + '\;' + term + '$' for par,term in zip(parameters, term_labels)]) + '\n$= 0$'


fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax = np.reshape(ax, [-1, ])
ax[0].plot(signals)
ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
ax[0].set_title(r'Model terms')

residuals_sq = np.sum(np.square(signals), axis=1)
ax[1].plot(residuals_sq)
ax[1].set_title(rf'Sum of squares of residuals: {Decimal(np.sum(residuals_sq)):.3E}' +
                '\nSquares of residuals')

term_energy = np.sum(np.square(signals), axis=0)
ax[2].bar(range(len(parameters)), term_energy,
          tick_label=term_labels)
ax[2].set_title(rf'Term energies')
ax[2].xaxis.set_tick_params(rotation=90)

ax[3].grid(False)
ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].text(0.2, -0.2, rf'{solution_string}', fontsize=12)

plt.show()

#%%
high_energy_idx = term_energy > 0.05*term_energy.max()

active_terms = active_terms[:, high_energy_idx]
term_labels = term_labels[high_energy_idx]
parameters = parameters[high_energy_idx]
signals = parameters*active_terms
solution_string = ' + \n'.join(['$' + str(round(par,3)) + '\;' + term + '$' for par,term in zip(parameters, term_labels)]) + '\n$= 0$'


fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax = np.reshape(ax, [-1, ])
ax[0].plot(signals)
ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
ax[0].set_title(r'Model terms')

residuals_sq = np.sum(np.square(signals), axis=1)
ax[1].plot(residuals_sq)
ax[1].set_title(rf'Sum of squares of residuals: {Decimal(np.sum(residuals_sq)):.3E}' +
                '\nSquares of residuals')

term_energy = np.sum(np.square(signals), axis=0)
ax[2].bar(range(len(parameters)), term_energy,
          tick_label=term_labels)
ax[2].set_title(rf'Term energies')
ax[2].xaxis.set_tick_params(rotation=90)

ax[3].grid(False)
ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].text(0.2, -0.2, rf'{solution_string}', fontsize=12)
