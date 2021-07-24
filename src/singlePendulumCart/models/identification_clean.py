import pandas as pd
import matplotlib.pyplot as plt
from src.utils.function_libraries import *
from src.utils.data_utils import *
from src.utils.identification.PI_Identifier import PI_Identifier
from src.utils.solution_processing import *
from differentiation.spectral_derivative import compute_spectral_derivative
from differentiation.finite_diff import compute_finite_diff
from filtering.SpectralFilter import SpectralFilter
from filtering.KernelFilter import KernelFilter
from tools import halve, mirror, add_noise, downsample
from src.utils.theta_processing.single_pend import *
from sklearn.model_selection import TimeSeriesSplit
import matplotlib as mpl
import os
import pickle
from containers.DynaFrame import DynaFrame, create_df
from definitions import ROOT_DIR
import sympy as sp
from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})

mpl.use('Qt5Agg')

datafile = 'singlePend.csv'
data_path = os.path.join(ROOT_DIR,'data','singlePend','simulated',datafile)
cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')

# Get training dataset
sim_data = pd.read_csv(data_path)
sim_data_x = sim_data.loc[:, ['x'+str(i) for i in (1,2)]]
sim_data_dx = sim_data.loc[:, ['dx'+str(i) for i in (1,2,3,4)]]
sim_data_u = sim_data.loc[:, 'u']
sim_data = pd.concat([sim_data_x, sim_data_dx, sim_data_u], axis=1)

sim_data = DynaFrame(sim_data)
dt = 0.001

# Get validation dataset
split_idx = int(len(sim_data)*0.8)
sim_data_val = DynaFrame(sim_data.iloc[split_idx:, :])
sim_data = DynaFrame(sim_data.iloc[:split_idx, :])

N = sim_data.shape[0]
step = 5

# Real signals
DX = sim_data.get_state_derivative_vars()
X = sim_data.get_state_vars()
X = pd.concat([X, DX.loc[:,['dx1', 'dx2']]], axis=1)
X.columns = ['x'+str(i) for i in (1,2,3,4)]
u = sim_data.get_input_vars()

# Validation data
DXval = sim_data_val.get_state_derivative_vars()
Xval = sim_data_val.get_state_vars()
Xval = pd.concat([Xval, DXval.loc[:,['dx1', 'dx2']]], axis=1)
Xval.columns = ['x'+str(i) for i in (1,2,3,4)]
uval = sim_data_val.get_input_vars()

#%%
u = downsample(u, step)
X = downsample(X, step)
X = create_df(X, 'x')
DX = downsample(DX, step)
DX = create_df(DX, 'dx')

Xval = downsample(Xval, step)
Xval = create_df(Xval, 'x')
DXval = downsample(DXval, step)
DXval = create_df(DXval, 'dx')
uval = downsample(uval, step)

#%%
data = {'X': X, 'DX': DX, 'u': u}
dataval = {'X': Xval, 'DX': DXval, 'u': uval}
theta_basis = create_basis(data)
theta_basis_val = create_basis(dataval)

theta_train = poly_library(theta_basis, (1,2,3,4))
theta_val = poly_library(theta_basis_val, (1,2,3,4))
# theta_validation = poly_library(theta_basis, (1,2,3,4))
#%%

theta_train = drop_bad_terms(theta_train)
theta_val = drop_bad_terms(theta_val)

theta_train.iloc[:,0] = 1
theta_train.iloc[0,0] = 1.00001

theta_val.iloc[:,0] = 1
theta_val.iloc[0,0] = 1.00001
# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'Clean'
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

    cachename = cache_str + '_' + eqn
    cachename = os.path.join(cache_path, cachename)

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat_train, theta_hat_validation)
        EqnIdentifier.set_thresh_range(lims=(0.0001, 0.05), n=10)
        EqnIdentifier.set_target(eqn)
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
    plot_implicit_sols(models, col_names, show_labels=False, axislabels=False)
    # %% Look for consistent models by finding clusters in the term activation space
    models = consistent_models(models, min_cluster_size=2)

    plot_implicit_sols(models, col_names, show_labels=False, axislabels=True)
    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    lhsvar = eqn_str
    # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    # Calculate AIC for each model
    models = model_aic(models, theta_val)
    # Drop obviously bad models
    aic_thresh = models['aic'].max() * 0.1
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
#%%
symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model3', symeqns),
        language='octave', to_files=True)

# 3 and 9
eqn_str = 'dx_3'
choice = 8
dxmodel = np.apply_along_axis(dynamic_model[eqn_str]['models'].loc[choice]['eqn_lambda'], axis=1, arr=sim_data)
dxreal = DXt.loc[:, eqn_str]
plt.figure()
plt.plot(dxreal, alpha=0.7, linestyle='--', color='tab:red')
plt.plot(dxmodel, alpha=0.7, color='tab:blue')
plt.legend(['Real', 'Model'])
plt.title('$\dot{x}_3$')
plt.xlabel('Sample index $k$')
plt.ylabel('Cart acceleration $\dot{x}_3$')
plt.show()

mdl = dynamic_model[eqn_str]['models'].loc[choice]
symeqn = sp.latex(mdl['eqn_sym'])

dx3 = dynamic_model['dx_3']['models'].loc[8]['eqn_sym']
dx4 = dynamic_model['dx_4']['models'].loc[6]['eqn_sym']

codegen(('identified_model2', [dx3, dx4]),
        language='octave', to_files=True)
