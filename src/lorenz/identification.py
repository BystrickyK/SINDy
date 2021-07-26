import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.function_libraries import *
from differentiation.spectral_derivative import compute_spectral_derivative
from filtering.SpectralFilter import SpectralFilter
from src.utils.identification.PI_Identifier import PI_Identifier
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

datafile = 'lorenz_sim_trig.csv'
data_path = os.path.join(ROOT_DIR,'data','lorenz',datafile)

# Get dataset
sim_data = pd.read_csv(data_path)
# sim_data = pd.concat([sim_data, sim_data[::-1]])
sim_data.rename(columns={sim_data.columns[0]: 't'}, inplace=True)
sim_data = DynaFrame(sim_data)
dt = sim_data.get_dt()

# Split the dataset into training and validation
sim_data, sim_data_test = train_test_split(sim_data, test_size=0.3, random_state=0, shuffle=False)

#%%
sim_data = DynaFrame(sim_data)
state_data = sim_data.get_state_vars()
state_derivative_data = sim_data.get_state_derivative_vars()

fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True)
for i in (0,1,2):
    axs[0, i].plot(state_data.iloc[50:-50, i], color='tab:blue')
    axs[1, i].plot(state_derivative_data.iloc[50:-50, i], color='tab:red')
    axs[0, i].set_title('x_' + str(i))
    plt.xlim([10, 20])
#%%
def create_theta(sim_data):
    sim_data = DynaFrame(sim_data)
    # Real signals
    input_data = sim_data.get_input_vars()
    state_data = sim_data.get_state_vars()
    state_derivative_data = sim_data.get_state_derivative_vars()
    trig_inputs = trigonometric_library(input_data)
    identification_data = pd.concat((input_data,
                               state_derivative_data,
                               state_data), axis=1)
    identification_data = DynaFrame(identification_data)

    theta1 = product_library(trig_inputs, identification_data).reset_index(drop=True)
    theta2 = poly_library(identification_data, (1,2)).reset_index(drop=True)
    theta = pd.concat([theta1, theta2], axis=1)
    return theta

theta_train = create_theta(sim_data)
theta_val = create_theta(sim_data_test)

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_1', 'dx_2', 'dx_3'] # State derivatives whose equation we want to identify
cache_str = 'Lorenz'
eqns_models = {}
for eqn in eqns_to_identify:
    print(eqn)

    idx = np.array([('d' in col and eqn not in col) for col in theta_train.columns])

    theta_train_eqn = theta_train.iloc[:, ~idx]
    theta_train_val = theta_train.iloc[:, ~idx]

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta_train_eqn
    eqns_models[eqn]['theta_val'] = theta_train_val

    # find the index of the target variable in the function library
    target_idx = np.argwhere(theta_train_eqn.columns == eqn)[0][0]

    # corr = theta.corr()
    # plot_corr(corr, theta.columns, labels=False, ticks=False)

    # EqnIdentifier = Explicit_Identifier(theta, target)
    EqnIdentifier = PI_Identifier(theta_train=theta_train_eqn, theta_val=theta_train_val, verbose=True)
    EqnIdentifier.set_thresh_range(lims=(0.0001, 0.01), n=10)
    EqnIdentifier.set_target(eqn)
    EqnIdentifier.set_guess_cols(target_idx)
    EqnIdentifier.create_models()
    eqns_models[eqn]['models'] = EqnIdentifier.models

# %%
dynamic_model = {}
for eqn_str, eqn_model in eqns_models.items():
    theta_train = eqn_model['theta_train']
    theta_val = eqn_model['theta_val']
    col_names = theta_train.columns
    models = eqn_model['models']
    dynamic_model[eqn_str] = {}

    models = model_unique(models)
    models = model_activations(models)

    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'u_1', 'u_2', 'u_3']
    lhsvar = eqn_str
    # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    # %%
    plot_implicit_sols(models, col_names, show_labels=True)
    plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = 0
    best_model = models.loc[choice]

    # %%
    dynamic_model[eqn_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[eqn_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[eqn_str]['models'] = models
    dynamic_model[eqn_str]['choice'] = best_model

    sim_data_test = DynaFrame(sim_data_test)
    sim_data_xu_test = pd.concat([sim_data_test.get_state_vars().reset_index(drop=True),
                                  sim_data_test.get_input_vars().reset_index(drop=True)],
                                 axis=1)
    sim_data_dx_test = sim_data_test.get_state_derivative_vars().reset_index(drop=True)

    dxmodel = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data_xu_test)
    dxreal = sim_data_dx_test.loc[:, eqn_str]

    plt.figure()
    plt.plot(dxmodel, alpha=0.8, color='tab:grey', linewidth=3)
    plt.plot(dxreal, '--', alpha=0.8, color='tab:blue', linewidth=2)
    plt.legend(['Model', 'Real'])
    plt.title(eqn_str)
    plt.show()
#%%
symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model3', symeqns),
        language='octave', to_files=True)
