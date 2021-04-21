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

X = Xc_train.values
DX = DXc_train.values
#%%
# state_data = X.x
N = X.shape[0]

step = 10

u_train = ForcingSignal(sim_data.iloc[:, -1], dt)
u_val = ForcingSignal(sim_data_val.iloc[:, -1], dt)

state_data_train = X.iloc[0:int(N/2):step, :].reset_index(drop=True)
state_derivative_data_train = DX.iloc[0:int(N/2):step, :].reset_index(drop=True)
input_data_train = u_train.values.iloc[0:int(N/2):step, :].reset_index(drop=True)
training_data = {'X':state_data_train, 'DX':state_derivative_data_train, 'u':input_data_train}

state_data_val = X.iloc[0:int(N/2):step, :].reset_index(drop=True)
state_derivative_data_val = DX.iloc[0:int(N/2):step, :].reset_index(drop=True)
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

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_3']  # State derivatives whose equation we want to identify
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
