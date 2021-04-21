import pandas as pd
import matplotlib.pyplot as plt
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.solution_processing import *
from utils.model_selection import *
import matplotlib as mpl
import os
import pickle

import sympy as sp
from sympy.utilities.codegen import codegen
from decimal import Decimal

# fft in moving window & averaging

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata_slow.csv'
filename_val = dirname + 'simdata_val.csv'

sim_data = pd.read_csv(filename)
sim_data = pd.concat([sim_data, sim_data[::-1]]).reset_index(drop=True)

sim_data_val = pd.read_csv(filename_val)
sim_data_val = pd.concat([sim_data_val, sim_data_val[::-1]]).reset_index(drop=True)

dt = sim_data.iloc[1, 0] - sim_data.iloc[0, 0]

# Generate a StateSignal object from the measurement data and add noise
rnp = 0.2*np.array([0.02, 0.0001, 0.05, 0.05])
Xm_train = StateSignal(sim_data.iloc[:, 1:-1], dt=dt, relative_noise_power=tuple(rnp))
Xm_val = StateSignal(sim_data_val.iloc[:, 1:-1], dt=dt, relative_noise_power=tuple(rnp))

# Real signals
Xc_train = StateSignal(Xm_train.values_clean, dt)  # Use clean data
DXc_train = StateDerivativeSignal(Xc_train, method='finitediff')

x = []
y = []
# for i in range(30):

# 0th order signals, positions
Xn_train = StateSignal(Xm_train.values.iloc[:, 0:2], dt)
filt = SpectralFilter(Xn_train)
gamma = 0.8
filt.find_cutoffs(k=0.98, gamma=gamma, plot=False)  # Find the bandwidth of strong frequencies
X_train = filt.filter(plot=False)  # Set all weaker frequencies to 0

Xn_val = StateSignal(Xm_val.values.iloc[:, 0:2], dt)
filt = SpectralFilter(Xn_val)
gamma = 0.8
filt.find_cutoffs(k=0.98, gamma=gamma, plot=False)  # Find the bandwidth of strong frequencies
X_val = filt.filter(plot=False)  # Set all weaker frequencies to 0

# 1st order signals, velocities
X_train = StateSignal(X_train.values, dt)
DX_train = StateDerivativeSignal(X_train, method='spectral', kernel_size=1)
X_val = StateSignal(X_val.values, dt)
DX_val = StateDerivativeSignal(X_val, method='spectral', kernel_size=1)

# 2nd order signals, accelerations
DDX_train = StateDerivativeSignal(DX_train, method='spectral', kernel_size=1)
DDX_train.values.columns = [f'dx_{i}' for i in [3, 4]]
DDX_val = StateDerivativeSignal(DX_val, method='spectral', kernel_size=1)
DDX_val.values.columns = [f'dx_{i}' for i in [3, 4]]
# DDXf = filt.filter(x=DDX.values.values, dt=dt, var_label='dx')
# DDXf.columns = [f'dx_{i}' for i in [3, 4]]

# Get full state vars / state var derivatives
N = int(sim_data.shape[0]/2)

X_full_train = pd.concat([X_train.values, DX_train.values], axis=1)
X_full_train = X_full_train.iloc[:N, :]
X_full_train.columns = [f'x_{i}' for i in [1,2,3,4]]

DX_full_train = pd.concat([DX_train.values, DDX_train.values], axis=1)
DX_full_train = DX_full_train.iloc[:N, :]
DX_full_train.columns = [f'dx_{i}' for i in [1,2,3,4]]

X_full_val = pd.concat([X_val.values, DX_val.values], axis=1)
X_full_val = X_full_val.iloc[:N, :]
X_full_val.columns = [f'x_{i}' for i in [1,2,3,4]]

DX_full_val = pd.concat([DX_val.values, DDX_val.values], axis=1)
DX_full_val = DX_full_val.iloc[:N, :]
DX_full_val.columns = [f'dx_{i}' for i in [1,2,3,4]]

# Gather the signals for easy plotting
X_train = X_train.values
X_c_train = Xc_train.values.iloc[:, 0:2]
X_n_train = Xm_train.values.iloc[:, 0:2]
DX_train = DX_train.values
# DX = DXf
DX_c_train = Xc_train.values.iloc[:, 2:4]
# DDX = DDXf
DDX_train = DDX_train.values
DDX_c_train = DXc_train.values.iloc[:, 2:4]

X_train.index = X_train.index*dt
X_c_train.index = X_c_train.index*dt
X_n_train.index = X_n_train.index*dt
DX_train.index = DX_train.index*dt
DX_c_train.index = DX_c_train.index*dt
DDX_train.index = DDX_train.index*dt
DDX_c_train.index = DDX_c_train.index*dt


# with plt.style.context(['seaborn-darkgrid', './images/BystrickyK.mplstyle']):
#     fig, axs = plt.subplots(nrows=X_train.shape[1], ncols=3, sharex=True, tight_layout=False)
#     left = 0
#     right = N//16
#     for i in range(axs.shape[0]):
#         axs[i,0].plot(X_train.iloc[left:right, i])
#         axs[i,0].plot(X_c_train.iloc[left:right, i])
#         axs[i,0].plot(X_n_train.iloc[left:right, i], linewidth=0.5, alpha=0.5)
#         axs[i,0].legend(["Spectral-Filtered", "Clean", "Noisy"],
#                         loc='lower right', fancybox=True, frameon=True)
#         axs[i,0].set_title(rf'$x_{i+1}$')
#         axs[i,1].plot(DX_train.iloc[left:right, i])
#         axs[i,1].plot(DX_c_train.iloc[left:right, i])
#         axs[i,1].legend(["Spectral-Diff", "Clean"],
#                         loc='lower right', fancybox=True, frameon=True)
#         axs[i,1].set_title(r'$\dot{xstr}_{sub}$'.format(xstr='x', sub=i+1))
#         axs[i,2].plot(DDX_train.iloc[left:right, i])
#         axs[i,2].plot(DDX_c_train.iloc[left:right, i])
#         axs[i,2].legend(["Spectral-DoubleDiff", "Clean"],
#                         loc='lower right', fancybox=True, frameon=True)
#         axs[i,2].set_title(r'$\ddot{xstr}_{sub}$'.format(xstr='x', sub=i+1))
#     plt.show()
#     [axs[X_train.shape[1]-1, i].set_xlabel(r'Time $t$ $[s]$') for i in [0,1,2]]
#
# plot_data = pd.concat([X_c_train, sim_data.iloc[:,-1]], axis=1)
# plot_data.index = plot_data.index * dt
# t = plot_data.index
# title_str = [r'Cart position $x_1$', r'Pendulum angle $x_2$', r'Force $u$']
# ylabels = [r'Position $x_1$ $[m]$', r'Angle $x_2$ $[rad]$', r'Force $u$ $[N]$']
# with plt.style.context(['seaborn-paper', './images/BystrickyK.mplstyle']):
#     fig, axs = plt.subplots(nrows=3, ncols=1, tight_layout=False, sharex=True)
#     for i, ax in enumerate(axs):
#         cm = plt.get_cmap()
#         if i==2:
#             ax.plot(t[left:right], plot_data.values[left:right, i], color='tab:red')
#         else:
#             ax.plot(t[left:right], plot_data.values[left:right, i], color='tab:blue')
#         # ax.set_title(title_str[i])
#         ax.set_xticks(range(9))
#         ax.set_ylabel(ylabels[i])
#     axs[-1].set_xlabel(r'Time $t$ $[s]$')
# axs[0].set_yticks(np.arange(-0.5, 0.5, 0.125))
# axs[1].set_yticks(np.arange(0,51,10))
# axs[2].set_yticks(np.arange(-400, 400, 100))

x_error = np.mean(np.sqrt(np.square(X_full_train-Xc_train.values)))
ddx_error = np.mean(np.sqrt(np.square(DX_full_train-DXc_train.values)))
print(f"Gamma: {gamma}\nMean error:\n{x_error}\n{ddx_error}\n")
#     x.append(gamma)
#     y.append(ddx_error[3])
# plt.figure()
# plt.semilogy(x, y, '-o')
# plt.show()

#%%
# state_data = X.x
N = X_full_train.values.shape[0]

step = 10

u_train = ForcingSignal(sim_data.iloc[:, -1], dt)
u_val = ForcingSignal(sim_data_val.iloc[:, -1], dt)

state_data_train = X_full_train.iloc[0:int(N/2):step, :].reset_index(drop=True)
state_derivative_data_train = DX_full_train.iloc[0:int(N/2):step, :].reset_index(drop=True)
input_data_train = u_train.values.iloc[0:int(N/2):step, :].reset_index(drop=True)
training_data = {'X':state_data_train, 'DX':state_derivative_data_train, 'u':input_data_train}

state_data_val = X_full_val.iloc[0:int(N/2):step, :].reset_index(drop=True)
state_derivative_data_val = DX_full_val.iloc[0:int(N/2):step, :].reset_index(drop=True)
input_data_val = u_val.values.iloc[0:int(N/2):step, :].reset_index(drop=True)
validation_data = {'X':state_data_val, 'DX':state_derivative_data_val, 'u':input_data_val}

dim = state_data_train.shape[1]

#%%
def create_basis(data):
    # identity = pd.Series((data['u'].values*0+1).T[0], data['u'].index, name='1')
    trig_basis_part = trigonometric_library(data['X'].iloc[:, 1])
    theta_basis = pd.concat([data['X'].iloc[:, [2,3]], trig_basis_part, data['u'],
                             data['DX'].iloc[:, [2,3]]], axis=1)
    return theta_basis

theta_basis_train = create_basis(training_data)
theta_basis_validation = create_basis(validation_data)

# scnd = product_library(theta_basis, theta_basis)
# thrd = product_library(scnd, theta_basis)
# frth = product_library(thrd, theta_basis)
# theta = pd.concat([theta_basis, scnd, thrd, frth, identity], axis=1)
theta_train = poly_library(theta_basis_train, (1,2,3,4))
theta_validation = poly_library(theta_basis_validation, (1,2,3,4))
#%%


# Remove terms which contain a single basis variable with a power higher than 2
# and terms which contain more than 3 basis terms
    # for example: x_3*x_3*x_3*x_1 contains x_3 to the third power -> is removed
    # or: sin(x_2)*sin(x_2)*sin(x_2) contains sin(x_2) to the third power -> is removed
    # or: sin(x_2)*cos(x_2)*u*x_4 contains 4 basis terms -> is removed
def drop_bad_terms(theta):
    bad_idx = np.array([False for i in theta.columns])
    for i, term in enumerate(theta.columns):
        multiplicands = term.split('*')
        unique_terms = list(set(multiplicands))
        unique_term_occurences = np.array([np.sum([term in mult for mult in multiplicands]) for term in unique_terms])
        terms_occurences = dict(zip(unique_terms, unique_term_occurences))
        if np.any(unique_term_occurences>2): # If any sub-term occurs more than two times
            bad_idx[i] = True
            continue
        if len(unique_terms)>3: # if there's more than 3 unique sub-terms in the term
            bad_idx[i] = True
            continue
        if (('x_3' in unique_terms and 'u' in unique_terms) or
            ('x_4' in unique_terms and 'u' in unique_terms) or
            ('x_4' in unique_terms and 'x_3' in unique_terms) or
            ('dx_3' in unique_terms and 'dx_4' in unique_terms)):
            bad_idx[i] = True
            continue
            # if sin(x_2) occurs more than once OR
            # if there are more than 2 trig sub-terms at once OR
            # if there are two or more occurences of u
        if ((terms_occurences.get('sin(x_2)', False))>1 or
                (terms_occurences.get('sin(x_2)', 0) + terms_occurences.get('cos(x_2)', 0))>2 or
                (terms_occurences.get('u', 0))>1):
            bad_idx[i] = True
            continue

    print(f'{np.sum(bad_idx)}/{len(theta.columns)})')
    theta = theta.iloc[:, ~bad_idx]
    return theta

theta_train = drop_bad_terms(theta_train)
theta_validation = drop_bad_terms(theta_validation)

theta_train.iloc[:,0] = 100
theta_validation.iloc[:,0] = 100
theta_train.iloc[0,0] = 100.001
theta_validation.iloc[0,0] = 100.001


# Plot first 6 and last 6 theta_i
# idxs = [*range(0,6), *range(theta_train.shape[1]-6, theta_train.shape[1])]
# axs = theta_train.iloc[0:5000, idxs].plot(subplots=True, layout=(4,3),
#                                     xlabel='$k$')
# axs = np.reshape(axs, [-1, ])
# [ax.legend( loc='upper center', fancybox=True, frameon=True) for ax in axs]
# idxs_str = ['{' + f'{idx}' + '}' for idx in idxs]
# [ax.set_title(f'$a_{idx}$') for idx,ax in zip(idxs_str, axs)]
# %%
# Build library with sums of angles (state var 2) and its sines/cosines
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
# bad_idx = np.array(['1*' in colname or 'sin(x_2)*sin(x_2)' in colname for colname in trig_bilinears.columns])
# trig_bilinears = trig_bilinears.iloc[:, ~bad_idx]
# # linear/angular accelerations -> second half of state var derivatives
# # %%
# theta = pd.concat([state_data, state_derivative_data, input_data, trig_data, trig_v_sq, trig_bilinears], axis=1)
# theta = theta.loc[:, ~theta.columns.duplicated()]
#
# dump_idx = np.array([col.count('sin') == 2 for col in theta.columns])  # Find indices of cols that contain sin^2(x)
# theta = theta.iloc[:, ~dump_idx]  # Keep all other columns

# Plot the correlation matrix of the regression matrix
# corr = theta_train.corr()
# plot_corr(corr, theta_train.columns, labels=False, ticks=False)
# plt.title("Training")
# # plt.show()
#
# corr = theta_validation.corr()
# plot_corr(corr, theta_validation.columns, labels=False, ticks=False)
# plt.title("Training")
# plt.show()

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'singlePendSolutionsNoisyDoubleDiff'
eqns_models = {}
for eqn in eqns_to_identify:
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta_train.columns])
    # Construct a library for identifying the desired equation
    theta_hat_train = theta_train.loc[:, ~idx]
    theta_hat_validation = theta_validation.loc[:, ~idx]
    # theta_hat_train = energy_normalize(theta_train.loc[:, ~idx])  # and keep all cols except them
    # theta_hat_validation = energy_normalize(theta_validation.loc[:, ~idx])  # and keep all cols except them

    # corr = theta_hat_train.corr()
    # plot_corr(corr, theta_hat_train.columns, labels=False, ticks=True)
    # plt.title(f"Correlation matrix of the function library for {eqn}")
    # plt.show()

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta_hat_train
    eqns_models[eqn]['theta_val'] = theta_hat_validation

    # svd = np.linalg.svd(theta_hat_train.values, full_matrices=False)
    # svd = {'U':svd[2], 'Sigma':np.diag(svd[1]), 'V':svd[0]}
    # plot_svd(svd)



    cachename = dirname + cache_str + '_' + eqn

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat_train, theta_hat_validation)
        EqnIdentifier.set_thresh_range(lims=(0.0000001, 0.1), n=10)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=7, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.all_models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)
# %%
dynamic_model = {}
for eqn_str, eqn_model in eqns_models.items():
    theta = eqn_model['theta_train']
    theta_val = eqn_model['theta_val']
    models = eqn_model['models']
    dynamic_model[eqn_str] = {}

    # %% Remove duplicate models
    models = process_models(models, theta.columns)
    models = models.reset_index(drop=True)
    # models = models.loc[models.valfit > 0.50, :].reset_index(drop=True)
    # plot_implicit_sols(models, theta.columns, show_labels=False)
    # Calculate RMSE for each model
    models['rmse_val'] = 0  # Create new column
    models['rmse_train'] = 0  # Create new column
    for idx, mdl in models.iterrows():
        rmse_train = calculate_rmse(theta, mdl['sol'], 0)
        models.loc[idx, 'rmse_train'] = rmse_train
        rmse_val = calculate_rmse(theta_val, mdl['sol'], 0)
        models.loc[idx, 'rmse_val'] = rmse_val

    # pareto_front(models, use_train=True, title=eqn_str)
    # pareto_front(models, use_train=False, title=eqn_str)

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

    # %% Look for consistent models by finding clusters in the term activation space
    models = consistent_models(models, dist,
                               min_cluster_size=3)

    plot_implicit_sols(models, theta.columns, show_labels=False)
    plt.show()
    # pareto_front(models)
    # plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models.valerror.argmin()
    model = models.iloc[choice, :]

    trainerror = model['trainerror']
    valerror = model['valerror']
    active_terms = theta.iloc[:, model['active']].values
    term_labels = theta.columns[model['active']]
    parameters = np.array(model['sol'])[model['active']]
    signals = parameters * active_terms
    solution_string = ' + \n'.join(
        ['$' + str(round(par, 3)) + '\;' + term + '$' for par, term in zip(parameters, term_labels)]) + '\n$= 0$'

    dynamic_model[eqn_str]['error'] = valerror
    dynamic_model[eqn_str]['term_labels'] = term_labels

    # fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    # ax = np.reshape(ax, [-1, ])
    # ax[0].plot(signals)
    # ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
    # ax[0].set_title(f'Errors: {round(trainerror, 4)}|{round(valerror, 4)}\nModel terms')
    #
    # residuals_sq = np.sum(np.square(signals), axis=1)
    # ax[1].plot(residuals_sq)
    # ax[1].set_title(rf'Sum of squares of residuals: {Decimal(np.sum(residuals_sq)):.3E}' +
    #                 '\nSquares of residuals')
    #
    # term_energy = np.sum(np.square(signals), axis=0)
    # ax[2].bar(range(len(parameters)), term_energy,
    #           tick_label=term_labels)
    # ax[2].set_title(rf'Term energies')
    # ax[2].xaxis.set_tick_params(rotation=90)
    #
    # ax[3].grid(False)
    # ax[3].set_xticklabels([])
    # ax[3].set_yticklabels([])
    # ax[3].text(0.2, -0.2, rf'{solution_string}', fontsize=12)
    #
    # plt.show()

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

    data = pd.concat([validation_data['X'], validation_data['u']], axis=1)
    eqn_data = data.apply(eqn_lambda, axis=1)

    print(f'Eqn: {eqn_str}\nEquation errors: {trainerror} | {valerror}\n')

symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model2', symeqns),
        language='octave', to_files=True)
