import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.solution_processing import *
import matplotlib as mpl
import os
import pickle
import sympy as sp
from sympy.utilities.codegen import codegen
from decimal import Decimal

# fft in moving window & averaging

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata_dt001.csv'

sim_data = pd.read_csv(filename)
sim_data = pd.concat([sim_data, sim_data[::-1]]).reset_index(drop=True)
# sim_data = pd.concat([sim_data, sim_data]).reset_index(drop=True)
# sim_data = pd.concat([sim_data, sim_data]).reset_index(drop=True)
# sim_data.loc[:, 'phi1'] /= (2*np.pi)
dt = sim_data.iloc[1, 0] - sim_data.iloc[0, 0]

# Create a filter object
Filter = KernelFilter(kernel='flattop', kernel_size=1)
# Generate a StateSignal object from the measurement data and add noise
rnp = 1*np.array([0.05, 0.0002, 0.05, 0.05])
X = StateSignal(sim_data.iloc[:, 1:-1], dt=dt, relative_noise_power=tuple(rnp))

Xn = StateSignal(X.values, dt)
Xc = StateSignal(X.values_clean, dt)  # Use clean data
# X = Filter.filter(X.values, 'x')  # Noisy data after filtering
DXc = StateDerivativeSignal(Xc, method='finitediff')

filt = SpectralFilter(Xn)
filt.find_cutoffs(k=0.95, gamma=1.3)  # Find the bandwidth of strong frequencies
Xf = filt.filter()  # Set all weaker frequencies to 0
filt = KernelFilter(kernel_size=1)  # Run the signal through a non-causal kernel filter
Xf = filt.filter(Xf.values)

# cutoff = 500
# Xc = cutoff_ends(Xc.values, cutoff)
# Xn = cutoff_ends(Xn.values, cutoff)
# Xf = cutoff_ends(Xf, cutoff)
# DXc = cutoff_ends(DXc.values, cutoff)

Xf = StateSignal(Xf.values, dt)
DX = StateDerivativeSignal(Xf, method='spectral', kernel_size=1)
DXfin = StateDerivativeSignal(X, method='finitediff')
u = ForcingSignal(sim_data.iloc[:, -1], dt)

Xf_error = np.array(Xf.values - Xc.values)
Xn_error = np.array(Xn.values - Xc.values)

# Xc = cutoff_ends(Xc, cutoff)
# Xn = cutoff_ends(Xn, cutoff)
# Xf = cutoff_ends(Xf.values, cutoff)
# DXc = cutoff_ends(DXc, cutoff)
# DX = cutoff_ends(DX.values, cutoff)
# u = cutoff_ends(u.values, 2*cutoff)

# filt = SpectralFilter(DX)
# filt.find_cutoffs(k=0.5, gamma=0.5, plot=True)
# DX = filt.filter()

fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, tight_layout=True)
left = 0
right = 58001
for i in range(axs.shape[0]):
    axs[i,0].plot(DXc.values.iloc[left:right, i], linewidth=2)
    axs[i,0].plot(DX.values.iloc[left:right, i], alpha=0.7)
    axs[i,0].legend(["Clean", "Spectral"])
    axs[i,1].plot(Xc.values.iloc[left:right, i], linewidth=2)
    axs[i,1].plot(Xn.values.iloc[left:right, i], alpha=0.7)
    axs[i,1].plot(Xf.values.iloc[left:right, i], alpha=0.7)
    axs[i,1].legend(["Clean", "Noisy", "Filtered"])
    axs[i,2].plot(Xf_error[left:right, i], alpha=0.7)
    axs[i,2].plot(Xn_error[left:right, i], alpha=0.7)
plt.show()
print(f"Mean square error: {np.sum(np.square(Xf_error), axis=0)/len(Xf_error)}")

# state_data = X.x
N = Xf.values.shape[0]

state_data = Xf.values.iloc[:int(N/2), :].reset_index(drop=True)
state_derivative_data = DX.values.iloc[:int(N/2), :].reset_index(drop=True)
input_data = u.values.iloc[:int(N/2), :].reset_index(drop=True)


# state_data = Xclean.x
# state_derivative_data = dXclean.dx
# input_data = u.u

dim = state_data.shape[1]
#%% Basis terms
trig_basis_part = trigonometric_library(state_data.iloc[:, 1])
theta_basis = pd.concat(state_data.iloc[:, (2,3)], trig_basis_part, input_data)

# state_derivative_data = state_derivative_data.iloc[:, dim//2:]  # Remove ambiguity between x3,x4 and dx1,dx2
# # %%
# # Build library with sums of angles (state var 2) and its sines/cosines
# trig_data = trigonometric_library(state_data.iloc[:, 1:dim // 2])
# trig_data = poly_library(trig_data, (1, 2))
#
# v_sq = square_library(state_data.iloc[:, 3:4])
#
# trig_v_sq = product_library(trig_data, v_sq)
# trig_v_sq = trig_v_sq.iloc[:, 1:]
#
# trig_bilinears = product_library(trig_data, pd.concat((state_data.loc[:, ('x_3', 'x_4')],
#                                                        state_derivative_data.loc[:, ('dx_3', 'dx_4')],
#                                                        input_data), axis=1))
#
# # because of cos^2(x) = 1-sin^2(x) identity
# bad_idx = np.array(['1*' in colname or 'sin(x_2)*sin(x_2)' in colname for colname in trig_bilinears.columns])
# trig_bilinears = trig_bilinears.iloc[:, ~bad_idx]
# # linear/angular accelerations -> second half of state var derivatives
# %%
# trig_state_derivative = product_library(trig_data, state_derivative_data)
# acceleration_data = trig_state_derivative

# Function library Theta
# theta = pd.concat([state_data, state_derivative_data,
#                    trig_data, trig_state_derivative], axis=1)
# theta = pd.concat([state_data, state_derivative_data, input_data, trig_data, trig_v_sq, trig_bilinears], axis=1)
# theta = theta.loc[:, ~theta.columns.duplicated()]

# keep_idx = [col.count('d')<2 and not 'x_1' in col for col in theta.columns]
# theta = theta.loc[:, keep_idx]
#
# x_count = [col.count('x') for col in theta.columns]
# trig = ['(' in col for col in theta.columns]
# dump_idx = np.array([x_c==3 and not trig for x_c,trig in zip(x_count, trig)])
# theta = theta.loc[:, ~dump_idx].reset_index()
# dump_idx = np.array([col.count('sin') == 2 for col in theta.columns])  # Find indices of cols that contain sin^2(x)
# theta = theta.iloc[:, ~dump_idx]  # Keep all other columns

# cutoff = 500
# theta = theta.iloc[cutoff:-cutoff, :]
# theta.reset_index(inplace=True, drop=True)

# theta.plot(subplots=True)
# theta.plot(subplots=True, layout=(3,4))
# Plot the correlation matrix of the regression matrix
corr = theta.corr()
plot_corr(corr, theta.columns, labels=False)
plt.show()

# %% Compute the solution or retrieve it from cache

rewrite = False # Should the cache be rewritten
eqns_to_identify = ['dx_1', 'dx_2', 'dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'singlePendSolutionsFullEnergyThreshNoisy'
eqns_models = {}
for eqn in eqns_to_identify:
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta.columns])
    # Construct a library for identifying the desired equation
    theta_hat = theta.loc[:, ~idx]  # and keep all cols except them

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta'] = theta_hat

    cachename = dirname + cache_str + '_' + eqn

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat)
        EqnIdentifier.set_thresh_range(lims=(0.00001, 0.1), n=10)
        EqnIdentifier.create_models(n_models=theta_hat.shape[1], iters=7, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.all_models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)
# %%
dynamic_model = {}
for eqn_str, eqn_model in eqns_models.items():
    theta = eqn_model['theta']
    models = eqn_model['models']
    dynamic_model[eqn_str] = {}

    # %% Remove duplicate models
    models = unique_models(models, theta.columns)
    models = models.loc[models.fit > 0.7, :].reset_index(drop=True)

    eqns = [[*zip(np.round(model.sol[model.active], 3), theta.columns[model.active])]
            for idx, model in models.iterrows()]
    eqns = [[str(par) + '*' + term for par, term in eqn] for eqn in eqns]
    eqns = [' + '.join(eqn) for eqn in eqns]

    idx = ['d' in eqn for eqn in eqns]
    models = models.loc[idx, :].reset_index()

    idx = []
    # %% Visualize the solutions -> calculate and plot activation distance matrix
    # and plot the matrix of implicit solutions
    dist = distance_matrix(models, plot=False)
    # plot_implicit_sols(models, theta.columns)
    # %% Look for consistent models by finding clusters in the term activation space
    models = consistent_models(models, dist,
                               min_cluster_size=2)

    # gsols = np.vstack(models.sol.values)
    # theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
    # gsols = gsols[:, theta_terms_idx]
    # glhs_guess_str = models.lhs.values

    plot_implicit_sols(models, theta.columns, show_labels=True)
    plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models.fit.argmax()
    model = models.iloc[choice, :]

    # Models with the same structure as the model with the best fit
    # choice_label = model['label']
    # good_models = models.loc[models['label'] == choice_label]
    # good_models = good_models.drop(['level_0', 'index'], axis=1).reset_index()
    # good_models = good_models.drop('index', axis=1)
    # for row in good_models.itertuples():
    #     signs = np.sign(row.sol)
    #     # If the signs of the current model correspond to the signs of the best model
    #     matching_signs = signs == np.sign(model.sol)
    #     if np.all(matching_signs):
    #         print(row.Index)
    #         pass
    #     else:
    #         print(f'oops {row.Index}')
    #         # Flip signs
    #         good_models.at[row.Index, 'sol'] = good_models.at[row.Index, 'sol'] * (-1)
    # # Average the models
    # model.sol = good_models.loc[:, 'sol'].mean()

    fit = model['fit']
    active_terms = theta.iloc[:, model['active']].values
    term_labels = theta.columns[model['active']]
    parameters = np.array(model['sol'])[model['active']]
    signals = parameters * active_terms
    solution_string = ' + \n'.join(
        ['$' + str(round(par, 3)) + '\;' + term + '$' for par, term in zip(parameters, term_labels)]) + '\n$= 0$'

    dynamic_model[eqn_str]['fit'] = fit

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

    # %%
    # Construct implicit function string
    eqn = [*zip(np.round(model.sol[model.active], 3), theta.columns[model.active])]
    eqn = [str(par) + '*' + term for par, term in eqn]
    eqn = ' + '.join(eqn)
    # Parse the string into a sympy expression
    symeqn = sp.parse_expr(eqn)
    symeqn = sp.solve(symeqn, eqn_str)[0]
    # Lambdify the sympy expression for evaluation
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    vars = [sp.parse_expr(var) for var in vars]
    lameqn = sp.lambdify(vars, symeqn)
    eqn_lambda = lambda row: lameqn(*row)  # one input in the form [x1, x2, ..., xn, u]
    dynamic_model[eqn_str]['lmbda'] = eqn_lambda
    dynamic_model[eqn_str]['symeqn'] = symeqn
    dynamic_model[eqn_str]['str'] = symeqn

    data = pd.concat([state_data, input_data], axis=1)
    eqn_data = data.apply(eqn_lambda, axis=1)
    # eqn_data = eqn_data[cutoff:-cutoff].reset_index()

    # fig, ax = plt.subplots(tight_layout=True)
    # ax.plot(eqn_data)
    # ax.plot(theta[eqn_str])
    # ax.legend(['Simulation data', 'Training data'])
    # ax.set_title(eqn_str)
    # plt.show()

    print(f'Eqn: {eqn_str}\nEquation fit: {fit}\n')

symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model2', symeqns),
        language='octave', to_files=True)
