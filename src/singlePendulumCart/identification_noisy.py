import pandas as pd
import matplotlib.pyplot as plt
from src.utils.identification.PI_Identifier import PI_Identifier
from src.utils.solution_processing import *
from differentiation.spectral_derivative import compute_spectral_derivative
from filtering.SpectralFilter import SpectralFilter
from tools import halve, mirror, add_noise, downsample
from src.utils.theta_processing.single_pend import *
from src.utils.visualization import *
import matplotlib as mpl
import os
from copy import copy
import pickle
from containers.DynaFrame import DynaFrame, create_df
from definitions import ROOT_DIR
import sympy as sp

from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})

mpl.use('Qt5Agg')

cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')

# Get training dataset
def load_data(data_path):
    sim_data = pd.read_csv(data_path)
    sim_data_x = sim_data.loc[:, ['s', 'phi1', 'Ds', 'Dphi']]
    sim_data_x.columns = ['x_' + str(i) for i in [1,2,3,4]]
    sim_data_dx = sim_data.loc[:, ['Ds', 'Dphi', 'DDs', 'DDphi']]
    sim_data_dx.columns = ['dx_' + str(i) for i in [1,2,3,4]]
    sim_data_u = sim_data.loc[:, 'u']
    sim_data_t = sim_data.loc[:, 't']
    sim_data = pd.concat([sim_data_t, sim_data_x, sim_data_dx, sim_data_u], axis=1)
    sim_data = DynaFrame(sim_data)
    dt = sim_data.get_dt()
    sim_data = sim_data.reset_index(drop=True)
    return DynaFrame(sim_data), dt

datafile = 'singlePend.csv'
data_path = os.path.join(ROOT_DIR,'data','singlePend','simulated',datafile)
sim_data, dt = load_data(data_path)
sim_data_clean = DynaFrame(copy(sim_data))

datafile = 'singlePend_val.csv'
data_path = os.path.join(ROOT_DIR,'data','singlePend','simulated',datafile)
sim_data_val, dt = load_data(data_path)
sim_data_val = DynaFrame(sim_data_val)


#%% Keep only the cart position and pendulum angle measurements and add noise
sim_data = sim_data.loc[:, ['x_1', 'x_2', 'u']]
# Add noise
sim_data.iloc[:,:] = add_noise(sim_data.values, [0.0005, 0.0025, 0])
sim_data_x_noisy = sim_data.loc[:, ['x_1', 'x_2']]
sim_data_u = sim_data.loc[:, 'u']

#%% Filter the measurements
filter = SpectralFilter(sim_data_x_noisy, dt, plot=True)
filter.find_cutoff_frequencies(offset=[-8, -8], std_thresh=15)
sim_data_x_filtered = filter.filter()

#%%
k = [19150, 19200]
compare_signals3(sim_data_clean.loc[k[0]:k[1], ['x_1', 'x_2']],
                 sim_data_x_noisy.iloc[k[0]:k[1], :],
                 sim_data_x_filtered.iloc[k[0]:k[1], :],
                 ['Clean','Noisy','Filtered'],
                 ['$x_1\ [m]$', '$x_2\ [rad]$'],
                 k=k[0])

#%%
sim_data_dx = compute_spectral_derivative(sim_data_x_filtered, dt, mirroring=True)
sim_data_dx = create_df(sim_data_dx, 'dx')

k = np.array(k) - np.array([2000, -2000])
# k = [0, len(sim_data_clean)]
compare_signals(sim_data_clean.loc[k[0]:k[1], ['dx_1', 'dx_2']],
                sim_data_dx.iloc[k[0]:k[1], :],
                ['Clean', 'Estimated'],
                ['$\dot{x}_1\ [\\frac{m}{s}]$', '$\dot{x}_2\ [\\frac{rad}{s}]$'],
                k=k[0])
#%%
sim_data_ddx = compute_spectral_derivative(sim_data_dx, dt, mirroring=True)
sim_data_ddx = create_df(sim_data_ddx, 'ddx')

compare_signals(sim_data_clean.loc[k[0]:k[1], ['dx_3', 'dx_4']],
                sim_data_ddx.iloc[k[0]:k[1], :],
                ['Clean', 'Estimated'],
                ['$\ddot{x}_1\ [\\frac{m}{s^2}]$', '$\ddot{x}_2\ [\\frac{rad}{s^2}]$'],
                k=k[0])


#%% Assemble training data
sim_data_train = pd.concat([sim_data_x_filtered,
                            sim_data_dx,
                            sim_data_ddx,
                            sim_data_u],
                           axis=1).reset_index(drop=True)
sim_data_train.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'dx_3', 'dx_4', 'u']

sim_data_train = cutoff(sim_data_train, 1250)
sim_data_train_full = copy(sim_data_train)

downsampling_step = 10
sim_data_train = downsample(sim_data_train, downsampling_step).reset_index(drop=True)
# sim_data_train = downsample(sim_data_clean, downsampling_step).reset_index(drop=True)

sim_data_train = DynaFrame(sim_data_train)

#%%
def data_dict(sim_data):
    data = {'X': sim_data.get_state_vars(),
            'DX': sim_data.get_state_derivative_vars(),
            'u': sim_data.get_input_vars()}
    return data

data_train = data_dict(sim_data_train)
data_val = data_dict(sim_data_val)

theta_basis = create_basis(data_train)
theta_basis_val = create_basis(data_val)

theta_train = poly_library(theta_basis, (1,2,3,4))
theta_val = poly_library(theta_basis_val, (1,2,3,4))

#%%

theta_train = drop_bad_terms(theta_train)
theta_val = drop_bad_terms(theta_val)

theta_train.iloc[:,0] = 1
theta_train.iloc[0,0] = 1.00001

theta_val.iloc[:,0] = 1
theta_val.iloc[0,0] = 1.00001

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
rewrite = False
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'SPFinalNoisy3'
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
        EqnIdentifier.set_thresh_range(lims=(0.0000001, 0.01), n=10)
        EqnIdentifier.set_target(eqn)
        # EqnIdentifier.set_guess_cols(guess_cols)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=8, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)

# %%
sim_data_xu = pd.concat([sim_data_val.get_state_vars(),
                           sim_data_val.get_input_vars()],
                          axis=1).reset_index(drop=True)
sim_data_dx = sim_data_val.get_state_derivative_vars().reset_index(drop=True)

dynamic_model = {}
for target_models_str, eqn_model in eqns_models.items():
    theta_train = eqn_model['theta_train']
    col_names = theta_train.columns
    theta_sub_val = theta_val.loc[:, col_names]
    models = eqn_model['models']
    dynamic_model[target_models_str] = {}

    step = 1
    #1
    print(f"#{step}: {len(models)}")
    step = step + 1

    # % Remove duplicate models
    models = model_unique(models)
    models = model_activations(models)
    models = model_val_rmse(models, theta_sub_val)
    # plot_implicit_sols(models, col_names, show_labels=False, axislabels=False)

    # Calculate AIC for each model
    models = model_aic(models, theta_sub_val)

    #%
    # model_metrics = models.loc[:, ['n_terms', 'train_metric', 'validation_metric', 'aic']]
    # model_metrics = model_metrics.sort_values('n_terms')
    # fig, axs = plt.subplots(ncols=2, tight_layout=True, sharex=True)
    # axs[0].plot(model_metrics['n_terms'], model_metrics['train_metric'], 'o',
    #             color='tab:blue', alpha=0.7)
    # axs[0].set_yscale('log')
    # axs[0].set_xlabel("$Number\ of\ terms$")
    # axs[0].set_ylabel("$Training\ RMSE$")
    #
    # axs[1].plot(model_metrics['n_terms'], model_metrics['validation_metric'],
    #             'o', color='tab:red', alpha=0.7)
    # axs[1].set_yscale('log')
    # axs[1].set_xlabel("$Number\ of\ terms$")
    # axs[1].set_ylabel("$Validation\ RMSE$")


    # % Look for consistent models by finding clusters in the term activation space
    models = model_consistent(models, min_cluster_size=2)

    #2
    print(f"#{step}: {len(models)}")
    step = step + 1

    # Discard non-sparse models
    models = model_sparse(models, threshold=10)

    #3
    print(f"#{step}: {len(models)}")
    step = step + 1

    plot_implicit_sols(models, col_names, show_labels=False, axislabels=True)
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
    aic_thresh = models['aic'].quantile(0.25) + 5
    models = models[ models['aic'] < aic_thresh ] # Keep models under the threshold

    #5
    print(f"#{step}: {len(models)}")
    step = step + 1

    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    #6
    print(f"#{step}: {len(models)}")
    step = step + 1

    # %
    plot_implicit_sols(models, col_names, show_labels=True)
    plt.show()

    # % Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models['aic'].argmin()
    best_model = models.loc[choice]

    # %
    dynamic_model[target_models_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[target_models_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[target_models_str]['models'] = models
    dynamic_model[target_models_str]['choice'] = best_model

    derivative_trajectory_model = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data_xu)
    derivative_trajectory_real = sim_data_dx.loc[:, target_models_str]

    dynamic_model[target_models_str]['model_val_traj'] = derivative_trajectory_model
    dynamic_model[target_models_str]['real_val_traj'] = derivative_trajectory_real

    #%%
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

fig = plt.figure(tight_layout=True, figsize=(10,8))
compare_signals(
    derivative_trajectory_real,
    derivative_trajectory_model,
    ['Reference model', 'Identified model'],
    ['$\\ddot{x}_1 = \\dot{x}_3\ [\\frac{m}{s^2}]$',
    '$\\ddot{x}_2 = \\dot{x}_4\ [\\frac{rad}{s^2}]$'])
plt.xlim([15000, 16000])

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
latex_output_file = 'model_latex.txt'
with open(latex_output_file, 'w') as file:
    file.write(latex_output)

model_name = 'identified_model_noisy'
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
models_cache_name = 'bestModelsNoisy'
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

