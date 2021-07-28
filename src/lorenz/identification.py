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
from fft.fft import fft
from fft.ifft import ifft
import pickle
from definitions import ROOT_DIR

import sympy as sp
from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({style_path, 'seaborn'})

mpl.use('Qt5Agg')

# datafile = 'lorenz_sim_trig.csv'
datafile = 'lorenz_sim_sgn.csv'
data_path = os.path.join(ROOT_DIR,'data','lorenz',datafile)

# Get dataset
sim_data = pd.read_csv(data_path)
# sim_data = pd.concat([sim_data, sim_data[::-1]])
sim_data.rename(columns={sim_data.columns[0]: 't'}, inplace=True)
sim_data = DynaFrame(sim_data)
dt = sim_data.get_dt()
time = range(len(sim_data)) * dt

#%%
fig = plt.figure(tight_layout=True, figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')
plot_lorentz3d_ax(np.array(sim_data.get_state_vars()), ax)
# ax.xaxis.label.set_fontsize(18)
# ax.yaxis.label.set_fontsize(18)
# ax.zaxis.label.set_fontsize(18)
#%%
data = sim_data.get_input_vars()
clr = 'tab:red'
labels = [r"$u_1$", r"$u_2$", r"$u_3$"]
fig, axs = plt.subplots(nrows=3, tight_layout=True, figsize=(12, 8), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(time, data.iloc[:, i], color=clr)
    ax.set_ylabel(labels[i])
axs[-1].set_xlabel(r"$Time\ t [s]$")
axs[0].set_title("Input measurements")

#%% Split the dataset into training and validation
sim_data, sim_data_test = train_test_split(sim_data, test_size=0.3, random_state=0, shuffle=False)

#%%
sim_data = DynaFrame(sim_data)
state_data = sim_data.get_state_vars()
state_derivative_data = sim_data.get_state_derivative_vars()

# fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True)
# for i in (0,1,2):
#     axs[0, i].plot(state_data.iloc[50:-50, i], color='tab:blue')
#     axs[1, i].plot(state_derivative_data.iloc[50:-50, i], color='tab:red')
#     axs[0, i].set_title('x_' + str(i))
#     plt.xlim([10, 20])



#%%
#%%
def create_theta(sim_data):
    sim_data = DynaFrame(sim_data)
    # Real signals
    input_data = sim_data.get_input_vars()
    state_data = sim_data.get_state_vars()
    state_derivative_data = sim_data.get_state_derivative_vars()

    trig_inputs = trigonometric_library(input_data)
    identification_data = pd.concat((input_data,
                               state_data), axis=1)
    identification_data = DynaFrame(identification_data)

    theta1 = product_library(trig_inputs, identification_data).reset_index(drop=True)
    theta2 = poly_library(identification_data, (1, 2)).reset_index(drop=True)
    trig_inputs.reset_index(drop=True, inplace=True)
    state_derivative_data.reset_index(drop=True, inplace=True)

    theta = pd.concat([theta1, theta2, trig_inputs, state_derivative_data], axis=1)
    return theta

theta = create_theta(sim_data)
theta['sgn(u_1)'] = np.sign(sim_data['u_1'].reset_index(drop=True))

# %% Compute the solution

eqns_to_identify = ['dx_1', 'dx_2', 'dx_3'] # State derivatives whose equation we want to identify
candidate_models_all = {}
for i, eqn in enumerate(eqns_to_identify):
    print(eqn)

    idx = np.array([('d' in col and eqn not in col) for col in theta.columns])

    theta_eqn = theta.iloc[:, ~idx]

    candidate_models_all[eqn] = {}
    candidate_models_all[eqn]['theta_cols'] = theta_eqn.columns

    # corr = theta.corr()
    # plot_corr(corr, theta.columns, labels=False, ticks=False)

    # EqnIdentifier = PI_Identifier(theta=theta_eqn, verbose=True)
    EqnIdentifier = PI_Identifier(theta=theta_eqn, verbose=True)
    EqnIdentifier.set_thresh_range(lims=(0.0001, 0.01), n=10)
    EqnIdentifier.set_target(eqn)
    EqnIdentifier.set_guess_cols(eqn)
    EqnIdentifier.create_models()
    candidate_models_all[eqn]['models'] = EqnIdentifier.models

# %%
dynamic_model = {}
for target_models_str, target_models in candidate_models_all.items():
    theta_cols = target_models['theta_cols']

    models = target_models['models']
    dynamic_model[target_models_str] = {}

    models = model_unique(models)

    models = model_activations(models)

    models = model_equation_strings(models, theta_cols)
    vars = ['x_1', 'x_2', 'x_3', 'u_1', 'u_2', 'u_3']
    models = model_symbolic_implicit_eqns(models, target_models_str)

    models = model_symbolic_eqn(models, target_models_str)
    models = model_lambdify_eqn(models, vars)

    models = models.reset_index(drop=True)

    plot_implicit_sols(models, theta_cols, show_labels=True)
    plt.show()

    # choice = int(input("Choose model index:"))
    choice = 0
    best_model = models.loc[choice]

    dynamic_model[target_models_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[target_models_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[target_models_str]['models'] = models
    dynamic_model[target_models_str]['choice'] = best_model

    sim_data_test = DynaFrame(sim_data_test)
    sim_data_xu_test = pd.concat([sim_data_test.get_state_vars().reset_index(drop=True),
                                  sim_data_test.get_input_vars().reset_index(drop=True)],
                                 axis=1)
    sim_data_dx_test = sim_data_test.get_state_derivative_vars().reset_index(drop=True)

    dx_model = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data_xu_test)
    dx_real = np.array(sim_data_dx_test.loc[:, target_models_str])

    dynamic_model[target_models_str]['model_val_traj'] = dx_model
    dynamic_model[target_models_str]['real_val_traj'] = dx_real

    # plt.figure()
    # plt.plot(dxmodel, alpha=0.8, color='tab:grey', linewidth=3)
    # plt.plot(dxreal, '--', alpha=0.8, color='tab:blue', linewidth=2)
    # plt.legend(['Model', 'Real'])
    # plt.title(target_models_str)
    # plt.show()

#%%
derivative_trajectory_real = []
derivative_trajectory_model = []
for eqn in eqns_to_identify:
    dx_traj_model = dynamic_model[eqn]['model_val_traj']
    dx_traj_real = dynamic_model[eqn]['real_val_traj']

    derivative_trajectory_model.append(dx_traj_model)
    derivative_trajectory_real.append(dx_traj_real)

derivative_trajectory_model = np.array(derivative_trajectory_model).T
derivative_trajectory_real = np.array(derivative_trajectory_real).T

fig = plt.figure(tight_layout=True, figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')
plot_lorentz3d_ax(derivative_trajectory_model, ax, 'Model', 'tab:blue')
plot_lorentz3d_ax(derivative_trajectory_real, ax, 'Real', 'tab:red', '--')
plt.legend()


#%%
def plot_periodogram(x, dt):
    omega, x_hat = fft(x, dt)

    x_psd = np.abs(x_hat)**1

    fig, axs = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(10,6))
    axs[0].plot(omega, x_psd, color='tab:blue')
    axs[0].set_xlabel("Frequency $\omega\ [\\frac{rad}{s}]$")
    axs[0].set_ylabel("Power $A_\omega$")
    axs[0].set_xlim([-200, 200])
    axs[1].plot(range(len(omega))*dt, x, color='tab:red')
    axs[1].set_xlabel("Time $t\ [s]$")
    axs[1].set_ylabel("Value $u_1$")
    plt.show()

u = np.array(sim_data.get_input_vars().iloc[:,0])
plot_periodogram(u, dt)
omega, x_hat = fft(u, dt)
#%%
symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
latex_output = ' \\\\ \n  '.join([sp.latex(eqn)  for eqn in symeqns])
latex_output_file = 'model_latex.txt'
with open(latex_output_file, 'w') as file:
    file.write(latex_output)


codegen(('identified_model3', symeqns),
        language='octave', to_files=True)