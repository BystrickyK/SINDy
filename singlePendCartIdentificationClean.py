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
filename = dirname + 'singlePend.csv'
filename_val = dirname + 'singlePend.csv'

# Get training dataset
sim_data = pd.read_csv(filename)
sim_data, dt = remove_time(sim_data)
# Append the mirrored version of the signal to deal with FFT Gibbs phenomena
sim_data = pd.concat([sim_data, sim_data[::-1]]).reset_index(drop=True)
sim_data = sim_data

N = sim_data.shape[0]
step = 1


# Real signals
Xt = sim_data.iloc[:, :-1]
Xt = create_df(Xt, 'x')

DXt = compute_spectral_derivative(Xt, dt)
DXt = create_df(DXt, 'dx')

DXt2 = compute_finite_differences(Xt, dt)
DXt2 = create_df(DXt2, 'dx')

u = sim_data.iloc[:, -1]
u = pd.DataFrame(u)
u.columns = ['u']
u = u.iloc[:N//2:step, :].reset_index(drop=True)

sim_data = sim_data.iloc[:N//2:step, :].values
Xt = Xt.iloc[:N//2:step, :].reset_index(drop=True)
DXt = DXt.iloc[:N//2:step, :].reset_index(drop=True)

# fig, axs = plt.subplots(nrows=2, tight_layout=True)
# axs[0].plot(Xt.iloc[:,2])
# axs[0].plot(DXt.iloc[:,0])
# axs[1].plot(Xt.iloc[:,3])
# axs[1].plot(DXt.iloc[:,1])
#%%
data = {'X': Xt, 'DX': DXt, 'u': u}
def create_basis(data):
    # identity = pd.Series((data['u'].values*0+1).T[0], data['u'].index, name='1')
    trig_basis_part = trigonometric_library(data['X'].iloc[:, 1])
    theta_basis = pd.concat([data['X'].iloc[:, [2,3]], trig_basis_part, data['u'],
                             data['DX'].iloc[:, [2,3]]], axis=1)
    return theta_basis

theta_basis = create_basis(data)

theta_train = poly_library(theta_basis, (1,2,3,4))
# theta_validation = poly_library(theta_basis, (1,2,3,4))
#%%

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
            ('dx_3' in unique_terms and 'u' in unique_terms) or
            ('dx_4' in unique_terms and 'u' in unique_terms) or
            ('dx_3' in unique_terms and 'dx_4' in unique_terms) or
            ('x_4' in unique_terms and 'x_3' in unique_terms)):
            bad_idx[i] = True
            continue
            # if sin(x_2) occurs more than once OR
            # if there are more than 2 trig sub-terms at once OR
            # if there are two or more occurences of u
        if ((terms_occurences.get('sin(x_2)', False))>1 or
                (terms_occurences.get('sin(x_2)', 0) + terms_occurences.get('cos(x_2)', 0))>2 or
                (terms_occurences.get('u', 0))>1 or
                (terms_occurences.get('dx_3', False))>1 or
                (terms_occurences.get('dx_4', False))>1):
            bad_idx[i] = True
            continue

    print(f'{np.sum(bad_idx)}/{len(theta.columns)})')
    theta = theta.iloc[:, ~bad_idx]
    return theta

theta_train = drop_bad_terms(theta_train)

theta_train.iloc[:,0] = 1
theta_train.iloc[0,0] = 1.00001

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'singlePendClean'
eqns_models = {}
for eqn in eqns_to_identify:
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta_train.columns])
    print(f'ii {np.sum(idx)}')

    # Construct a library for identifying the desired equation
    theta_hat_train = theta_train.loc[:, ~idx]
    theta_hat_validation = theta_train.loc[:, ~idx]

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta_hat_train
    eqns_models[eqn]['theta_val'] = theta_hat_validation

    # corr = theta_hat_train.corr()
    # plot_corr(corr, theta_hat_train.columns, labels=False, ticks=True)

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
        EqnIdentifier.set_thresh_range(lims=(0.00001, 0.025), n=5)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=8, shuffle=False)
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
    models = consistent_models(models, min_cluster_size=4)

    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    lhsvar = eqn_str
    # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    # Calculate AIC for each model
    models = model_aic(models, theta_val)
    # Drop obviously bad models
    aic_thresh = models['aic'].max() * 0.5
    models = models[ models['aic'] < aic_thresh ] # Keep models under the threshold

    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    # %%
    plot_implicit_sols(models, col_names, show_labels=False)
    plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models['aic'].argmin()
    best_model = models.loc[choice]

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
    dynamic_model[eqn_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[eqn_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[eqn_str]['models'] = models
    dynamic_model[eqn_str]['choice'] = best_model

    # data = pd.concat([validation_data['X'], validation_data['u']], axis=1)
    # eqn_data = data.apply(eqn_lambda, axis=1)
    #
    # print(f'Eqn: {eqn_str}\nEquation errors: {trainerror} | {valerror}\n')

    dxmodel = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data)
    dxreal = DXt.loc[:, eqn_str]

    plt.figure()
    plt.plot(dxmodel, alpha=0.8)
    plt.plot(dxreal, alpha=0.8)
    plt.legend(['Model', 'Real'])
    plt.title(eqn_str)
    plt.show()

symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model2', symeqns),
        language='octave', to_files=True)
