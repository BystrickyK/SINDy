import pandas as pd
import matplotlib.pyplot as plt
from src.utils.identification.PI_Identifier import PI_Identifier
from src.utils.solution_processing import *
from differentiation.spectral_derivative import compute_spectral_derivative
from filtering.SpectralFilter import SpectralFilter
from tools import halve, mirror, add_noise, downsample
from src.utils.theta_processing.single_pend import *
from sklearn.model_selection import train_test_split
from src.utils.visualization import *
import matplotlib as mpl
import os
from copy import copy
import pickle
from containers.DynaFrame import DynaFrame, create_df
from definitions import ROOT_DIR
import sympy as sp

from sympy.utilities.codegen import codegen

mpl.use('Qt5Agg')

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})


cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')


datafile = 'processed_measurements.csv'
data_path = os.path.join(ROOT_DIR,'data','singlePend','real',datafile)
data = pd.read_csv(data_path)
data['u'] = data['u'] - data['u'].mean()
data = DynaFrame(data)
dt = data.get_dt()

#%%
data_x = data.loc[:, ['x_1', 'x_2', 'x_3', 'x_4', 'u']]

#%% Filter the measurements
filter = SpectralFilter(data_x, dt, plot=True)
filter.find_cutoff_frequencies(offset=[0, 0, -2, -2, 0], std_thresh=2000)
data_x_filtered = filter.filter()

#%%
compare_signals(data_x, data_x_filtered,
                 ['Measured', 'Filtered'],
                 ['$x_1\ [\mathrm{m}]$', '$x_2\ [\mathrm{rad}]$',
                  '$x_3\ [\\frac{\mathrm{m}}{\mathrm{s}}]$', '$x_4\ [\\frac{\mathrm{rad}}{\mathrm{s}}]$',
                  '$u\ [-]$'])

data_u = data_x_filtered.loc[:, 'u']
data_positions = data_x_filtered.loc[:, ['x_1', 'x_2']]
data_velocities = data_x_filtered.loc[:, ['x_3', 'x_4']]
#%%
data_accelerations = compute_spectral_derivative(data_velocities, dt)
plot_signals(data_accelerations, ['Cart $\dot{x}_3\ [\\frac{\mathrm{m}}{\mathrm{s}^2}]$', 'Pendulum $\dot{x}_4\ [\\frac{\mathrm{rad}}{\mathrm{s}^2}]$'])
data_accelerations = create_df(data_accelerations)
data_accelerations.columns = ['dx_3', 'dx_4']

#%% Assemble training data
sim_data_train = pd.concat([data_positions, data_velocities,
                            data_u, data_accelerations],
                           axis=1).reset_index(drop=True)
sim_data_train.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'u', 'dx_3', 'dx_4']

sim_data_train = cutoff(sim_data_train, 1000)

sim_data = copy(sim_data_train)
# sim_data_train, sim_data_test = train_test_split(sim_data_train, test_size=0.3,
#                                                  random_state=42, shuffle=True)

downsampling_step = 10
sim_data_train = downsample(sim_data_train, downsampling_step).reset_index(drop=True)
# sim_data_train = downsample(sim_data_clean, downsampling_step).reset_index(drop=True)

sim_data_train = DynaFrame(sim_data_train)
# sim_data_test = DynaFrame(sim_data_test)

#%%
def data_dict(sim_data):
    data = {'X': sim_data.get_state_vars(),
            'DX': sim_data.get_state_derivative_vars(),
            'u': sim_data.get_input_vars()}
    return data

data_train = data_dict(sim_data_train)
# data_test = data_dict(sim_data_test)

theta_basis = create_basis(data_train)
# theta_basis_test = create_basis(data_test)

theta_train = poly_library(theta_basis, (1,2,3))
# theta_val = poly_library(theta_basis_test, (1,2,3))

theta_train['x_4*x_4*sin(x_2)*cos(x_2)'] = theta_train['x_4'] * theta_train['x_4'] \
                                           * theta_train['sin(x_2)'] * theta_train['cos(x_2)']

# theta_val['x_4*x_4*sin(x_2)*cos(x_2)'] = theta_val['x_4'] + theta_val['x_4'] \
#                                            + theta_val['sin(x_2)'] + theta_val['cos(x_2)']
#%%

theta_train = drop_bad_terms(theta_train)
# theta_val = drop_bad_terms(theta_val)

theta_train.iloc[:,0] = 1
theta_train.iloc[0,0] = 1.00001

# theta_val.iloc[:,0] = 1
# theta_val.iloc[0,0] = 1.00001
# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
rewrite = False
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'SPFinalReal2'
eqns_models = {}
for i, eqn in enumerate(eqns_to_identify):
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta_train.columns])
    print(f'ii {np.sum(idx)}')

    # Construct a library for identifying the desired equation
    theta_hat_train = theta_train.loc[:, ~idx]

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta_train'] = theta_hat_train

    # corr = theta_hat_train.corr()
    # plot_corr(corr, theta_hat_train.columns, labels=False, ticks=True)

    cachename = cache_str + '_' + eqn
    cachename = os.path.join(cache_path, cachename)

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        cache_path = os.path.join(ROOT_DIR, 'src', 'singlePendulumCart', 'cache')
        guess_cache_name = 'guessColumnsReal'
        guess_cache_path = os.path.join(cache_path, guess_cache_name)
        with open(guess_cache_path, 'rb') as f:
            guess_cols = pickle.load(f)[i]
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat_train)
        EqnIdentifier.set_thresh_range(lims=(0.0000001, 0.1), n=30)
        EqnIdentifier.set_target(eqn)
        # EqnIdentifier.set_guess_cols(guess_cols)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=8, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)

# %%
sim_data_test = DynaFrame(sim_data)
sim_data_xu_test = pd.concat([sim_data_test.get_state_vars().reset_index(drop=True),
                              sim_data_test.get_input_vars().reset_index(drop=True)],
                             axis=1)
sim_data_dx_test = sim_data_test.get_state_derivative_vars().reset_index(drop=True)

dynamic_model = {}
for target_models_str, eqn_model in eqns_models.items():
    theta_train = eqn_model['theta_train']
    col_names = theta_train.columns
    models = eqn_model['models']
    dynamic_model[target_models_str] = {}

    step = 1
    #1
    print(f"#{step}: {len(models)}")
    step = step + 1

    # % Remove duplicate models
    models = model_unique(models)
    models = model_activations(models)
    # plot_implicit_sols(models, col_names, show_labels=False, axislabels=False)

    # % Look for consistent models by finding clusters in the term activation space
    # models = model_consistent(models, min_cluster_size=2)

    #2
    print(f"#{step}: {len(models)}")
    step = step + 1

    # Discard non-sparse models
    models = model_sparse(models, threshold=10)

    #3
    print(f"#{step}: {len(models)}")
    step = step + 1

    # plot_implicit_sols(models, col_names, show_labels=False, axislabels=True)
    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    lhsvar = target_models_str
    # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    #4
    print(f"#{step}: {len(models)}")
    step = step + 1

    #%
    # Drop bad models
    rmse_thresh = models['train_metric'].quantile(0.25)
    models = models[ models['train_metric'] < rmse_thresh ] # Keep models under the threshold

    #5
    print(f"#{step}: {len(models)}")
    step = step + 1

    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    #6
    print(f"#{step}: {len(models)}")
    step = step + 1

    models = model_validate(models, sim_data_xu_test, sim_data_dx_test.loc[:, target_models_str])
    # %
    # plot_implicit_sols(models, col_names, show_labels=True, aic=True)
    plt.show()

    rmse_thresh = models['rmse'].quantile(0.25)
    models = models[ models['rmse'] < rmse_thresh ] # Keep models under the threshold
    models.reset_index(drop=True, inplace=True)

    plot_implicit_sols(models, col_names, show_labels=True, aic=True)
    plt.show()
    # % Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models['rmse'].argmin()
    best_model = models.loc[choice]

    #7
    print(f"#{step}: {len(models)}")
    step = step + 1
    # %
    dynamic_model[target_models_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[target_models_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[target_models_str]['models'] = models
    dynamic_model[target_models_str]['choice'] = best_model


    derivative_trajectory_model = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data_xu_test)
    derivative_trajectory_real = sim_data_dx_test.loc[:, target_models_str]

    dynamic_model[target_models_str]['model_val_traj'] = derivative_trajectory_model
    dynamic_model[target_models_str]['real_val_traj'] = derivative_trajectory_real

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

compare_signals(
    derivative_trajectory_real,
    derivative_trajectory_model,
    ['Reference', 'Model predictions'],
    ['$\\ddot{x}_1 = \\dot{x}_3\ [\\frac{m}{s^2}]$',
    '$\\ddot{x}_2 = \\dot{x}_4\ [\\frac{rad}{s^2}]$'])


errors = derivative_trajectory_real - derivative_trajectory_model
plot_signals(errors, ['$\\epsilon_1$', '$\\epsilon_2$', '$\\epsilon_3$'])
err_str = [str(e) for e in np.round(np.sqrt(np.mean(np.square(errors), axis=0)), 5)]
print(err_str)

#%% Autogenerate model code for a MATLAB ODE function
def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})
symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
symeqns = [round_expr(sp.simplify(sp.factor(eqn)), 5) for eqn in symeqns]

# os.chdir('..')
latex_output = ' \\\\ \n  '.join([sp.latex(eqn)  for eqn in symeqns])
latex_output_file = 'model_latex_real.txt'
with open(latex_output_file, 'w') as file:
    file.write(latex_output)

model_name = 'identified_model_real'
os.chdir('models')
codegen((model_name, symeqns),
        language='octave', to_files=True)

#%% Extract best models and save their structure into cache
keys = dynamic_model.keys()
best_models = [dynamic_model[key]['choice'] for key in keys]

model_structures = []
for key, model in zip(keys, best_models):
    model_structure = {'active': model['active'],
                      'target': key,
                      'col_names': eqns_models[key]['theta_train'].columns}
    model_structures.append(model_structure)

cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')
models_cache_name = 'bestModelsReal'
models_cache_path = os.path.join(cache_path, models_cache_name)
with open(models_cache_path, 'wb') as f:
    pickle.dump(model_structures, f)

#%% Save good guess columns in cache
# good_guesses = []
# for eqn, results in dynamic_model.items():
#     print(eqn)
#     models = results['models']
#     active_cols = models['active'].values
#     active_cols = np.vstack(active_cols)
#     active_cols = active_cols.any(axis=0)
#     good_guesses.append(active_cols)
# good_guesses = np.array(good_guesses)
# # good_guesses = good_guesses.any(axis=0)
# # good_guesses = np.argwhere(good_guesses).T[0]
# good_guesses = [np.argwhere(g).T[0] for g in good_guesses]
#
# cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')
# guess_cache_name = 'guessColumns'
# guess_cache_path = os.path.join(cache_path, guess_cache_name)
# with open(guess_cache_name, 'wb') as f:
#     pickle.dump(good_guesses, f)
#%%

