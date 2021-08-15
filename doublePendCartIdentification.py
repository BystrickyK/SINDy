import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.solution_processing import *
from utils.model_selection import *
from utils.theta_processing_dPend import *
import matplotlib as mpl
import os
import pickle

import sympy as sp
from sympy.utilities.codegen import codegen
from decimal import Decimal

# fft in moving window & averaging

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'doublePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'doublePendSimData.csv'

# Get training dataset
sim_data = pd.read_csv(filename)
sim_data, dt = remove_time(sim_data)
# Append the mirrored version of the signal to deal with FFT Gibbs phenomena
X = sim_data.iloc[:, [0,1,2,3,4,5]]
DX = sim_data.iloc[:, [3,4,5,6,7,8]]
u = sim_data.iloc[:,-1]

valsimdata = pd.concat([X, u], axis=1).reset_index(drop=True)

N = sim_data.shape[0]
step = 100

# Real signals
Xt = X.iloc[::step].values
Xt = create_df(Xt, 'x')

DXt = DX.iloc[::step].values
DXt = create_df(DXt, 'dx')

u = u.iloc[::step].reset_index(drop=True)
#%%
data = {'X': Xt, 'DX': DXt, 'u': u}
theta_basis = create_basis(data)

theta_train = poly_library(theta_basis, (1,2,3,4,5,6))
#%%

theta_train = drop_bad_terms(theta_train)

theta_train.iloc[:,0] = 1
theta_train.iloc[0,0] = 1.00001

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_4', 'dx_5', 'dx_6']  # State derivatives whose equation we want to identify
cache_str = 'doublePendSolClean1'
eqns_models = {}
for i, eqn in enumerate(eqns_to_identify):
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta_train.columns])
    print(f'ii {np.sum(idx)}')

    # Construct a library for identifying the desired equation
    theta_hat_train = theta_train.loc[:, ~idx]
    theta_hat_validation = theta_train.loc[:, ~idx]

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta_hat_train
    eqns_models[eqn]['theta_val'] = theta_hat_validation

    weights = theta_train[eqn]
    weights = weights.abs() + 1
    weights = weights.max() - weights
    weights = weights - weights.min() + 1
    weights = weights / weights.max()
    f = KernelFilter()
    weights = f.filter(weights)
    weights = np.power(weights, 1.5)

    cachename = dirname + cache_str + '_' + eqn

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        guess_cache_name = 'guessColumnsReal'
        with open(guess_cache_name, 'rb') as f:
            guess_cols = pickle.load(f)[i]
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat_train, theta_hat_validation)
        EqnIdentifier.set_thresh_range(lims=(0.000001, 0.01), n=3)
        EqnIdentifier.set_weights(weights)
        EqnIdentifier.set_target(eqn)
        EqnIdentifier.set_guess_cols(guess_cols)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=3, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)

# %%
dynamic_model = {}
for eqn_str, eqn_model in eqns_models.items():
    theta_train = eqn_model['theta_train']
    theta_val = eqn_model['theta_val']
    col_names = theta_train.columns
    models = eqn_model['models']
    dynamic_model[eqn_str] = {}

    # %% Remove duplicate models
    models = unique_models(models)
    models = model_activations(models)
    # %% Look for consistent models by finding clusters in the term activation space
    models = consistent_models(models, min_cluster_size=3, distance_threshold=1)
    print(f"Number of models left 1: {len(models)}")

    # metric_thresh = 0.2
    models = model_aic(models, theta_val)
    models.sort_values(by='aic', axis=0, inplace=True)
    # keep_percent = 0.1  # What fraction of models should be kept
    # models = models.iloc[:int(np.floor(keep_percent*len(models))), :]
    keep_num = 5  # How many models should be kept
    models = models.iloc[:keep_num, :]
    # models = models[ models['val_metric'] < metric_thresh ]

    print(f"Number of models left 2: {len(models)}")

    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'u']
    lhsvar = eqn_str
    # # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    print(f"Number of models left 3: {len(models)}")
    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    # %%
    plot_implicit_sols(models, col_names, show_labels=False, axislabels=True)
    plt.show()

    # %% Decompose one of the models
    choice = models['aic'].argmin()
    best_model = models.loc[choice]

    # %%
    dynamic_model[eqn_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[eqn_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[eqn_str]['models'] = models
    dynamic_model[eqn_str]['choice'] = best_model

#%%
def plot_choice(choice, eqn_str, ylabel=None):
    dxmodel = np.apply_along_axis(dynamic_model[eqn_str]['models'].loc[choice]['eqn_lambda'], axis=1, arr=valsimdata.loc[::step,:])
    dxreal = DXt.loc[:, eqn_str]
    plt.figure()
    plt.plot(dxreal, alpha=0.7, linestyle='--', color='tab:red')
    plt.plot(dxmodel, alpha=0.7, color='tab:blue')
    plt.legend(['Real', 'Model'])
    # plt.title('$\dot{x}_4$')
    plt.title(ylabel)
    plt.xlabel('Sample index $k$')
    # plt.ylabel('Pendulum acceleration $\dot{x}_4$')
    plt.ylabel(ylabel)
    plt.show()

plot_choice(0, 'dx_4', 'Pendulum acceleration $\dot{x}_4$')
plot_choice(0, 'dx_5', 'First pendulum acceleration $\dot{x}_5$')
plot_choice(0, 'dx_6', 'Second pendulum acceleration $\dot{x}_6$')
#%%
# mdl = dynamic_model[eqn_str]['models'].loc[choice]
# symeqn = sp.latex(mdl['eqn_sym'])
#
# dx4 = dynamic_model['dx_4']['models'].loc[0]['eqn_sym']
# dx5 = dynamic_model['dx_5']['models'].loc[0]['eqn_sym']
# dx6 = dynamic_model['dx_6']['models'].loc[0]['eqn_sym']
# symeqns = [dx4, dx5, dx6]
symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
#
codegen(('doublePendIdentified', symeqns),
        language='octave', to_files=True)

#%%
good_guesses = []
for eqn, results in dynamic_model.items():
    print(eqn)
    models = results['models']
    active_cols = models['active'].values
    active_cols = np.vstack(active_cols)
    active_cols = active_cols.any(axis=0)
    good_guesses.append(active_cols)
good_guesses = np.array(good_guesses)
# good_guesses = good_guesses.any(axis=0)
# good_guesses = np.argwhere(good_guesses).T[0]
good_guesses = [np.argwhere(g).T[0] for g in good_guesses]

guess_cache_name = 'guessColumnsReal'
with open(guess_cache_name, 'wb') as f:
    pickle.dump(good_guesses, f)
