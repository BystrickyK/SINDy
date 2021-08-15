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
from sklearn.model_selection import train_test_split
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

sim_data, dt = load_data(data_path)
sim_data, sim_data_test = train_test_split(sim_data, test_size=0.2,
                                            shuffle=False, random_state=42)

#%%
# dx = compute_spectral_derivative(x, dt, mirroring=True)
# dx = create_df(dx, 'dx')
# filter = KernelFilter(kernel_size=51)
# dx = filter.filter(dx)
# compare_signals(DXt, dx, ['Clean', 'Filtered'], ylabels=['$\dot{x}_1 \; [m\; s^{-2}]$',
#                                                           '$\dot{x}_2 \; [rad\; s^{-2}]$'])
#
# ddx = compute_spectral_derivative(dx, dt)
# ddx = create_df(ddx, 'ddx')
# compare_signals(DDXt, ddx, ['Clean', 'Filtered'], ylabels=['$\ddot{x}_1 \; [m\; s^{-2}]$',
#                                                                          '$\ddot{x}_2 \; [rad\; s^{-2}]$'])
#%% Downsample training data
sim_data = downsample(sim_data, 10).reset_index(drop=True)
sim_data = DynaFrame(sim_data)
sim_data_test = DynaFrame(sim_data_test)

# compare_signals(DX.iloc[:,[2,3]], downsample(DDXt.iloc[:,[0,1]], step),
#                 legend_str=['Filt','Clean'], ylabels=['a', 'b'])
#%%
def data_dict(sim_data):
    data = {'X': sim_data.get_state_vars(),
            'DX': sim_data.get_state_derivative_vars(),
            'u': sim_data.get_input_vars()}
    return data

data = data_dict(sim_data)
data_val = data_dict(sim_data_test)

theta_basis = create_basis(data)
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
cache_str = 'SPFinalDense'
eqns_models = {}
for eqn in eqns_to_identify:
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
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat_train)
        EqnIdentifier.set_thresh_range(lims=(0.000001, 0.01), n=5)
        EqnIdentifier.set_target(eqn)
        EqnIdentifier.create_models(n_models=theta_hat_train.shape[1], iters=8, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)

# %%
sim_data_xu = pd.concat([sim_data_test.get_state_vars(),
                           sim_data_test.get_input_vars()],
                          axis=1).reset_index(drop=True)
sim_data_dx = sim_data_test.get_state_derivative_vars().reset_index(drop=True)

dynamic_model = {}
for target_models_str, eqn_model in eqns_models.items():
    theta_train = eqn_model['theta_train']
    col_names = theta_train.columns
    theta_sub_val = theta_val.loc[:, col_names]
    models = eqn_model['models']
    dynamic_model[target_models_str] = {}

    # %% Remove duplicate models
    models = model_unique(models)
    models = model_activations(models)
    models = model_val_rmse(models, theta_sub_val)
    # plot_implicit_sols(models, col_names, show_labels=False, axislabels=False)

    # Calculate AIC for each model
    models = model_aic(models, theta_sub_val)

    #%%
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


    # %% Look for consistent models by finding clusters in the term activation space
    models = model_consistent(models, min_cluster_size=2)

    # Discard non-sparse models
    models = model_sparse(models, threshold=10)

    # plot_implicit_sols(models, col_names, show_labels=False, axislabels=True)
    models = model_equation_strings(models, col_names)
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    lhsvar = target_models_str
    # Create symbolic implicit equations column
    models = model_symbolic_implicit_eqns(models, lhsvar)

    #%%
    # Drop bad models
    aic_thresh = models['aic'].max() * 0.1
    models = models[ models['aic'] < aic_thresh ] # Keep models under the threshold

    models = model_symbolic_eqn(models, lhsvar)
    models = model_lambdify_eqn(models, vars)
    models = models.reset_index(drop=True)

    # %%
    plot_implicit_sols(models, col_names, show_labels=True)
    plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models['aic'].argmin()
    best_model = models.loc[choice]

    # %%
    dynamic_model[target_models_str]['symeqn'] = best_model['eqn_sym']
    dynamic_model[target_models_str]['str'] = best_model['eqn_sym_implicit']
    dynamic_model[target_models_str]['models'] = models
    dynamic_model[target_models_str]['choice'] = best_model

    derivative_trajectory_model = np.apply_along_axis(best_model['eqn_lambda'], axis=1, arr=sim_data_xu)
    derivative_trajectory_real = sim_data_dx.loc[:, target_models_str]

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

# fig = plt.figure(tight_layout=True, figsize=(9,8))
compare_signals(derivative_trajectory_real, derivative_trajectory_model,
                ['Real', 'Model'], ['$\\dot{x_3}$', '$\\dot{x_4}$'])

#%%
def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
symeqns = [round_expr(sp.simplify(sp.factor(eqn)), 5) for eqn in symeqns]

latex_output = ' \\\\ \n  '.join([sp.latex(eqn)  for eqn in symeqns])
latex_output_file = 'model_latex.txt'
with open(latex_output_file, 'w') as file:
    file.write(latex_output)

os.chdir('models')
codegen(('identified_model_clean', symeqns),
        language='octave', to_files=True)

#%%
sim_data = DynaFrame(sim_data)
plot_signals(sim_data.get_state_vars(),
            # ['$\\dot{x_3}$', '$\\dot{x_4}$']
            ['$x_1\ [m]$',
             '$x_2\ [rad]$',
             '$x_3=\\dot{x}_1\ [\\frac{m}{s}]$',
             '$x_4=\\dot{x}_2\ [\\frac{rad}{s}]$'])

#%% Save good guess columns
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


cache_path = os.path.join(ROOT_DIR,'src', 'singlePendulumCart', 'cache')
guess_cache_name = 'guessColumnsReal'
guess_cache_path = os.path.join(cache_path, guess_cache_name)
with open(guess_cache_path, 'wb') as f:
    pickle.dump(good_guesses, f)
