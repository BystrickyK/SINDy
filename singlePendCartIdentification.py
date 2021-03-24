import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import *
from utils.solution_processing import *
import matplotlib as mpl
import os
import pickle
import sympy as sp
from decimal import Decimal

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata.csv'

sim_data = pd.read_csv(filename)
dt = sim_data.iloc[1, 0] - sim_data.iloc[0, 0]

Filter = KernelFilter(kernel='hann', kernel_size=31)
X = StateSignal(sim_data.iloc[:, 1:-1], dt=dt, relative_noise_power=(0.02, 0.01, 0.02, 0.01))
# X = Filter.filter(X.values, 'x')  # Noisy data after filtering
X = X.values_clean  # Use clean data

X = StateSignal(X, dt)
DX = StateDerivativeSignal(X, method='spectral')
u = ForcingSignal(sim_data.iloc[:, -1], dt)

# state_data = X.x
state_data = X.values
state_derivative_data = DX.values
input_data = u.values


# state_data = Xclean.x
# state_derivative_data = dXclean.dx
# input_data = u.u

dim = state_data.shape[1]

state_derivative_data = state_derivative_data.iloc[:, dim//2:]  # Remove ambiguity between x3,x4 and dx1,dx2
#%%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
trig_data = trigonometric_library(state_data.iloc[:, 1:dim//2])

# linear/angular accelerations -> second half of state var derivatives

#%%
trig_state_derivative = product_library(trig_data, state_derivative_data)
acceleration_data = trig_state_derivative

# Function library Theta
theta = pd.concat([state_data, state_derivative_data,
                   trig_data, trig_state_derivative], axis=1)

cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]
# theta.plot(subplots=True, layout=(3,4))

# Plot the correlation matrix of the regression matrix
corr = theta.corr()
plot_corr(corr, theta.columns)
plt.show()

#%% Compute the solution or retrieve it from cache
cachename = dirname + 'singlePendSolutionsEnergyThresh'
rewrite = False
if os.path.exists(cachename) and not rewrite:
    print("Retrieving solution from cache.")
    with open(cachename, 'rb') as f:
        models = pickle.load(f)
else:
    print("No solution in cache, calculating solution from scratch.")
    EqnIdentifier = PI_Identifier(theta)
    EqnIdentifier.set_thresh_range(lims=(0.0001, 0.5), n=10)
    EqnIdentifier.create_models(n_models=theta.shape[1], iters=7, shuffle=False)
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
                           min_cluster_size=3)

# gsols = np.vstack(models.sol.values)
# theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
# gsols = gsols[:, theta_terms_idx]
# glhs_guess_str = models.lhs.values

plot_implicit_sols(models, theta.columns, show_labels=False)

#%% Decompose one of the models
choice = 8
model = models.iloc[choice, :]

fit = model['fit']
active_terms = theta.iloc[:, model['active']].values
term_labels = theta.columns[model['active']]
parameters = np.array(model['sol'])[model['active']]
signals = parameters*active_terms
solution_string = ' + \n'.join(['$' + str(round(par,3)) + '\;' + term + '$' for par,term in zip(parameters, term_labels)]) + '\n$= 0$'


fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax = np.reshape(ax, [-1, ])
ax[0].plot(signals)
ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
ax[0].set_title(f'Fit: {round(fit, 4)}\nModel terms')

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
high_energy_idx = term_energy > 0.1*term_energy.max()

active_terms = active_terms[:, high_energy_idx]
term_labels = term_labels[high_energy_idx]
parameters = parameters[high_energy_idx]
signals = parameters*active_terms
solution_string = ' + \n'.join(['$' + str(round(par,3)) + '\;' + term + '$' for par,term in zip(parameters, term_labels)]) + '\n$= 0$'


fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax = np.reshape(ax, [-1, ])
ax[0].plot(signals)
ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
ax[0].set_title(f'Model terms')

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
#%%
# Construct implicit function string
eqn = [*zip(np.round(model.sol[model.active], 3), theta.columns[model.active])]
eqn = [str(par) + '*' + term for par,term in eqn]
eqn = ' + '.join(eqn)
eqn = eqn.replace('dx_3', 'u')  # dx_3 == u b
# Parse the string into a sympy expression
symeqn = sp.parse_expr(eqn)
symeqn = sp.solve(symeqn, 'dx_4')
# Lambdify the sympy expression for evaluation
vars = [('x_2', 'x_4'), 'u']
lameqn = sp.lambdify(vars, symeqn)
