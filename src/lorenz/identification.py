import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.function_libraries import *
from differentiation.spectral_derivative import compute_spectral_derivative
from filtering.SpectralFilter import SpectralFilter
from src.utils.identification.Explicit_Identifier import Explicit_Identifier
from solution_processing import *
from model_selection import *
from theta_processing import *
from sklearn.model_selection import train_test_split
from containers.DynaFrame import DynaFrame, create_df
from tools import halve, mirror
import matplotlib as mpl
import os
import pickle
from definitions import ROOT_DIR

import sympy as sp
from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})

mpl.use('Qt5Agg')

datafile = 'lorenz_sim.csv'
data_path = os.path.join(ROOT_DIR,'data','lorenz',datafile)

# dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
# filename = dirname + 'singlePend.csv'
# filename_val = dirname + 'singlePend.csv'

# Get training dataset
sim_data = pd.read_csv(data_path)
# sim_data = pd.concat([sim_data, sim_data[::-1]])
sim_data.rename(columns={sim_data.columns[0]: 't'}, inplace=True)
sim_data = DynaFrame(sim_data)
dt = sim_data.get_dt()

# Real signals
state_data = sim_data.get_state_vars()
state_data_filtered = state_data

state_derivative_data = sim_data.get_state_derivative_vars()
# Filter state data
# filter = SpectralFilter(state_data, dt, plot=False)
# filter.find_cutoff_frequencies()
# state_data_filtered = filter.filter()
# compare_signals(state_data, state_data_filtered,
#                 ('Noisy', 'Filtered'),
#                 (" ", " ", ""))
# plt.show()

# state_derivative_data = compute_spectral_derivative(mirror(state_data_filtered), dt)
# state_derivative_data = create_df(halve(state_derivative_data), 'dx')

#%%
fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True)
for i in (0,1,2):
    axs[0, i].plot(state_data_filtered.iloc[50:-50, i], color='tab:blue')
    axs[1, i].plot(state_derivative_data.iloc[50:-50, i], color='tab:red')
    axs[0, i].set_title('x_' + str(i))
#%%
sim_data.reset_index(inplace=True, drop=True)
state_data_filtered.reset_index(inplace=True, drop=True)
state_derivative_data.reset_index(inplace=True, drop=True)
identification_data = pd.concat((sim_data.get_input_vars(),
                           state_derivative_data,
                           state_data_filtered), axis=1)
identification_data = identification_data.iloc[500:-500, :]
# identification_data, testing_data = train_test_split(identification_data,
#                                                      test_size=0.3,
#                                                      random_state=32,
#                                                      shuffle=False)
identification_data = DynaFrame(identification_data)

tmp = pd.concat([identification_data.get_state_vars(),
                 identification_data.get_input_vars()], axis=1)
# tmp = identification_data.get_state_vars()
theta = poly_library(tmp, (1,2,3))
# theta, _ = remove_twins(theta)
targets = identification_data.get_state_derivative_vars()

# testing_data = DynaFrame(testing_data)
# x_u_test = pd.concat([testing_data.get_state_vars(), testing_data.get_input_vars()], axis=1)
# dx_test = testing_data.get_state_derivative_vars()
# x_u_test.reset_index(inplace=True, drop=True)
# dx_test.reset_index(inplace=True, drop=True)
# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = targets.columns[1:2] # State derivatives whose equation we want to identify
cache_str = 'Lorenz'
eqns_models = {}
for eqn in eqns_to_identify:
    print(eqn)

    target = targets.loc[:, eqn]
    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta
    eqns_models[eqn]['theta_val'] = theta

    # corr = theta.corr()
    # plot_corr(corr, theta.columns, labels=False, ticks=False)

    EqnIdentifier = Explicit_Identifier(theta, target)
    EqnIdentifier.set_thresh_range(lims=(0.001, 1), n=3)
    EqnIdentifier.create_models()
    eqns_models[eqn]['models'] = EqnIdentifier.models

#%%
models = eqns_models['dx_2']
models = models['models']

# %%
# dynamic_model = {}
# for eqn_str, eqn_model in eqns_models.items():
#     theta_train = eqn_model['theta_train']
#     theta_val = eqn_model['theta_val']
#     col_names = theta_train.columns
#     models = eqn_model['models']
#     dynamic_model[eqn_str] = {}
#
#     # %% Remove duplicate models
#     models = unique_models(models)
#     models = model_activations(models)
#     # %% Look for consistent models by finding clusters in the term activation space
#     # models = consistent_models(models, min_cluster_size=2)
#
#     models = model_equation_strings(models, col_names)
#     vars = ['x_1', 'x_2', 'x_3', 'u_1', 'u_2', 'u_3']
#     lhsvar = eqn_str
#     # Create symbolic implicit equations column
#     models = model_symbolic_implicit_eqns(models, lhsvar)
#
#     # Calculate AIC for each model
#     models = model_aic(models, theta_val)
#     # Drop obviously bad models
#     aic_thresh = models['aic'].max() * 0.1
#     models = models[ models['aic'] < aic_thresh ] # Keep models under the threshold
#
#     models = model_symbolic_eqn(models, lhsvar)
#     models = model_lambdify_eqn(models, vars)
#     models = models.reset_index(drop=True)
#
#     # %%
#     plot_implicit_sols(models, col_names, show_labels=True)
#     plt.show()
#
#     # %% Decompose one of the models
#     # choice = int(input("Choose model index:"))
#     choice = models['aic'].argmin()
#     best_model = models.loc[choice]
#
#     # %%
#     dynamic_model[eqn_str]['symeqn'] = best_model['eqn_sym']
#     dynamic_model[eqn_str]['str'] = best_model['eqn_sym_implicit']
#     dynamic_model[eqn_str]['models'] = models
#     dynamic_model[eqn_str]['choice'] = best_model
#
#     dxmodel = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=x_u_test)
#     dxreal = dx_test.loc[:, eqn_str]
#
#     plt.figure()
#     plt.plot(dxmodel, alpha=0.8)
#     plt.plot(dxreal, alpha=0.8)
#     plt.legend(['Model', 'Real'])
#     plt.title(eqn_str)
#     plt.show()
# #%%
# symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
# codegen(('identified_model3', symeqns),
#         language='octave', to_files=True)
