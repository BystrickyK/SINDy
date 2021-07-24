import pandas as pd
import matplotlib.pyplot as plt
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.solution_processing import *
from utils.model_selection import *
from utils.theta_processing_sPend import *
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

# Get validation dataset
sim_data_val = pd.read_csv(filename_val)
sim_data_val, dt = remove_time(sim_data_val)
# Append the mirrored version of the signal to deal with FFT Gibbs phenomena
sim_data_val = pd.concat([sim_data_val, sim_data_val[::-1]]).reset_index(drop=True)

N = sim_data.shape[0]
step = 5

# Real signals
Xt = sim_data.iloc[:, :-1]
Xt = create_df(Xt, 'x')

DXt = compute_spectral_derivative(Xt, dt)
DXt = create_df(DXt, 'dx')

DXt2 = compute_finite_differences(Xt, dt)
DXt2 = create_df(DXt2, 'dx')

# Validation data
Xval = sim_data_val.iloc[:, :-1]
Xval = create_df(Xval, 'x')
DXval = compute_spectral_derivative(Xval, dt)
DXval = create_df(DXval, 'dx')
uval = sim_data_val.iloc[:, -1]
uval = pd.DataFrame(uval)
uval.columns = ['u']

# xn = add_noise(sim_data.iloc[:, [0, 1]], [0.0025, 0.005])
xn = add_noise(sim_data.iloc[:, [0, 1]], [0.0025, 0.005])
filter = SpectralFilter(xn, dt, plot=False)
filter.find_cutoffs_and_meanlogpower(k=0.98, freq_multiplier=1)
# x2 = filter.decrease_modulus()
x = filter.filter(var_label='x')

with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
    fig, axs = plt.subplots(nrows=2, tight_layout=True, sharex=True)
    axs[0].plot(xn.iloc[:, 0], alpha=0.7, linewidth=2, color='tab:red')
    axs[0].plot(x.iloc[:, 0], alpha=1, linewidth=2, color='tab:blue')
    axs[0].plot(Xt.iloc[:, 0], alpha=0.8, linewidth=2, color='tab:green')
    axs[0].set_ylabel('$x_1\; [m]$')
    axs[0].legend(['Noisy', 'Clean', 'Filtered'])

    axs[1].plot(xn.iloc[:, 1], alpha=0.7, linewidth=2, color='tab:red')
    axs[1].plot(x.iloc[:, 1], alpha=1, linewidth=2, color='tab:blue')
    axs[1].plot(Xt.iloc[:, 1], alpha=0.8, linewidth=2, color='tab:green')
    axs[1].set_ylabel('$x_2 \; [rad]$')
    axs[1].legend(['Noisy', 'Clean', 'Filtered'])
    axs[1].set_xlabel('Sample index $k$')

dx = compute_spectral_derivative(x, dt)
# dx = compute_finite_differences(x, dt)
# filter = KernelFilter(kernel_size=50)
# dx = filter.filter(dx)
dx = create_df(dx, 'dx')

# compare_signals(DXt.iloc[:, [0,1]], dx, ['Clean', 'Filtered'], varlabel='\dot{x}')

ddx = compute_spectral_derivative(dx, dt)
ddx = create_df(ddx, 'ddx')
compare_signals(DXt.iloc[:, [2,3]], ddx, ['Clean', 'Filtered'], ylabels=['$\ddot{x}_1 \; [m\; s^{-2}]$',
                                                                         '$\ddot{x}_2 \; [rad\; s^{-2}]$'])

u = sim_data.iloc[:, -1]
u = pd.DataFrame(u)
u.columns = ['u']
u = u.iloc[:N//2-20:step, :].reset_index(drop=True)

sim_data = sim_data.iloc[:N//2:step, :].values

X = np.array(pd.concat([x, dx], axis=1))
X = create_df(X, 'x')

DX = np.array(pd.concat([dx, ddx], axis=1))
DX = create_df(DX, 'dx')

X = X.iloc[:N//2-20:step, :].reset_index(drop=True)
DX = DX.iloc[:N//2-20:step, :].reset_index(drop=True)

Xt = Xt.iloc[:N//2:step, :].reset_index(drop=True)
DXt = DXt.iloc[:N//2:step, :].reset_index(drop=True)

Xval = Xval.iloc[:N//2:step, :].reset_index(drop=True)
DXval = DXval.iloc[:N//2:step, :].reset_index(drop=True)
uval = uval.iloc[:N//2:step, :].reset_index(drop=True)
# compare_signals(DX.iloc[:,[2,3]], DXt.iloc[:,[2,3]], legend_str=['Filt','Clean'])
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

rewrite = False # Should the cache be rewritten
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'STC2'
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
        EqnIdentifier.set_thresh_range(lims=(0.0001, 0.05), n=10)
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
    models = consistent_models(models, min_cluster_size=2)

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
