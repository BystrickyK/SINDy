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
from library_creator import LibraryCreator
from sklearn.linear_model import Ridge
from tools import d_to_dot
from copy import copy, deepcopy
import pickle
import seaborn as sb
from containers.DynaFrame import DynaFrame, create_df
from definitions import ROOT_DIR
import sympy as sp

def d_to_dot2(str_):
    return d_to_dot([str_]) + '_'
from sympy.utilities.codegen import codegen

style_path = os.path.join(ROOT_DIR, 'src', 'utils', 'visualization', 'BystrickyK.mplstyle')
print(style_path)
plt.style.use({'seaborn', style_path})

mpl.use('Qt5Agg')

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
# compare_signals(data_x, data_x_filtered,
#                 ['Measured', 'Filtered'],
#                 ['$x_1\ [\mathrm{m}]$', '$x_2\ [\mathrm{rad}]$',
#                  '$x_3\ [\\frac{\mathrm{m}}{\mathrm{s}}]$', '$x_4\ [\\frac{\mathrm{rad}}{\mathrm{s}}]$',
#                  '$u\ [-]$'])

data_u = data_x_filtered.loc[:, 'u']
data_positions = data_x_filtered.loc[:, ['x_1', 'x_2']]
data_velocities = data_x_filtered.loc[:, ['x_3', 'x_4']]
#%%
data_accelerations = compute_spectral_derivative(data_velocities, dt)
# plot_signals(data_accelerations, ['Cart $\dot{x}_3\ [\\frac{\mathrm{m}}{\mathrm{s}^2}]$', 'Pendulum $\dot{x}_4\ [\\frac{\mathrm{rad}}{\mathrm{s}^2}]$'])
data_accelerations = create_df(data_accelerations)
data_accelerations.columns = ['dx_3', 'dx_4']

#%% Assemble training data
sim_data_train = pd.concat([data_positions, data_velocities,
                            data_u, data_accelerations],
                           axis=1).reset_index(drop=True)
sim_data_train.columns = ['x_1', 'x_2', 'x_3', 'x_4', 'u', 'dx_3', 'dx_4']

#%% Load the reference structure models
model_cache_name = 'bestModelsReal'
model_path = os.path.join(cache_path, model_cache_name)
with open(model_path, 'rb') as f:
    model_structures = pickle.load(f)

sim_data_train = cutoff(sim_data_train, 1250)

# sim_data_train = downsample(sim_data_train, downsampling_step).reset_index(drop=True)
sim_data_train = DynaFrame(sim_data_train)


#%% Create the nonlinear data libraries from measurements
targets = ([mdl['target'] for mdl in model_structures])
active_cols = [mdl['col_names'][mdl['active']] for mdl in model_structures]

modeldict = {key: {} for key in targets}

for model in model_structures:
    target = model['target']
    active = model['active']
    print(target)
    function_strings = np.array(model['col_names'][active])

    Lib = LibraryCreator(target, function_strings, sim_data_train)
    datalib = Lib.create_library()
    sym_vars = Lib.symvars

    modeldict[target]['library'] = datalib


#%%
n_models = 10000
for i, target in enumerate(targets):
    A = modeldict[target]['library']
    target_idx = np.argwhere(A.columns == target)[0][0]

    # col_choice = np.random.choice(len(A.columns), 1)[0]
    # print(f"{A.columns[col_choice]}")
    # list of parameter vectors
    X = []
    RHS = []

    for n in range(n_models):
        # Sample from the training dataset
        random_idx = np.random.choice(A.index, len(A)//10)
        Aw = A.iloc[random_idx, :].reset_index(drop=True)  # resampled library

        col_choice = np.random.choice(len(Aw.columns), 1)[0]
        bw = Aw.iloc[:, col_choice]
        Aw = Aw.drop(Aw.columns[col_choice], axis=1)

        model = Ridge(alpha=1, fit_intercept=False)
        model.fit(Aw.values, bw.values)
        x = list(model.coef_)
        x.insert(col_choice, -1)
        x = np.array(x)
        x = x / x[target_idx]
        X.append(x)
        RHS.append(col_choice)

        if np.mod(n, 100) == 0:
            print(n)

    df_dict = {'x': X,
               'b': RHS}
    param_df = pd.DataFrame.from_dict(df_dict)

    modeldict[target]['models'] = X
    modeldict[target]['rhs'] = RHS
    modeldict[target]['param_df'] = param_df

#%%
def splatexify(parse_str):
    return sp.latex(sp.parse_expr(parse_str))

#%%
for i, target in enumerate(targets):
    df = modeldict[target]['param_df']
    df['b_str'] = active_cols[i][df['b']]
    modeldict[target]['param_df'] = df

#%%
cmap = plt.cm.tab10(np.linspace(0,1,10))
for i, target in enumerate(targets):
    X = np.array(modeldict[target]['models'])
    rhs = np.array(modeldict[target]['rhs'])
    term_strs = modeldict[target]['library'].columns

    n_terms = len(term_strs)
    fig, axs = plt.subplots(nrows=n_terms//2, ncols=2, tight_layout=True, figsize=(12, 10))
    axs = axs.reshape(-1, )

    modes = []
    # for choice in range(0, n_terms):
    #     Xw = X[rhs==choice]
    for col in range(X.shape[1]):
        hist = axs[col].hist(X[:, col], bins=100, alpha=0.8,
                             color='tab:blue')
        axs[col].set_title(f'Parameter: {latexify([term_strs[col]])[0]}', fontsize=16)
        axs[col].locator_params(nbins=12, axis='x')
        axs[col].tick_params(axis='x', size=5, labelsize=10, labelrotation=30)

        mode_idx = np.argmax(hist[0])
        mode = hist[1][mode_idx]
        axs[col].vlines(mode, 0, hist[0][mode_idx]*0.5, color='black', linewidth=2)
        modes.append(mode)
    # axs[-1].legend(latexify(term_strs), fontsize=14, loc='upper left',
    #               bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), facecolor='white', framealpha=1)
    modeldict[target]['best_params'] = modes

#%%
def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

#%%
vars = [sp.parse_expr(var) for var in ['x_1', 'x_2', 'x_3', 'x_4', 'u']]
symeqns = []
for target in targets:
    pars = modeldict[target]['best_params']
    pars = [str(round(par, 6)) for par in pars]
    fun_strs = modeldict[target]['library'].columns

    terms_strs = ['*'.join([par, fun]) for par,fun in zip(pars, fun_strs)]
    implicit_str = ' + '.join(terms_strs)
    implicit_str = implicit_str

    symeqn = sp.parse_expr(implicit_str)
    targetvar = sp.parse_expr(target)
    sol = sp.solve(symeqn, targetvar)[0]
    # explicit_symeqn = sp.solve()
    symeqns.append(sol)

#%%
symeqns = [round_expr(sp.simplify(sp.factor(eqn)), 5) for eqn in symeqns]

# os.chdir('..')
latex_output = ' \\\\ \n  '.join([sp.latex(eqn)  for eqn in symeqns])
latex_output_file = 'model_latex_real_bs.txt'
with open(latex_output_file, 'w') as file:
    file.write(latex_output)

model_name = 'identified_model_real_bs'
os.chdir('models')
codegen((model_name, symeqns),
        language='octave', to_files=True)

#%%
def lambdify(symeqn):
    eqn_lambda_ = sp.lambdify(vars, symeqn, {'sgn': np.sign})
    eqn_lambda = lambda input: eqn_lambda_(*input)
    return eqn_lambda

sim_data_xu = pd.concat([sim_data_train.get_state_vars(), sim_data_train.get_input_vars()], axis=1).reset_index(drop=True)
sim_data_dx = sim_data_train.get_state_derivative_vars()

lambdas = []
dx_model = []
dx_real = []
for target, symeqn in zip(targets, symeqns):
    eqn_lambda = lambdify(symeqn)
    derivative_trajectory_model = np.apply_along_axis(eqn_lambda, axis=1, arr=sim_data_xu)
    derivative_trajectory_real = sim_data_dx.loc[:, target]
    dx_model.append(derivative_trajectory_model)
    dx_real.append(derivative_trajectory_real)
    lambdas.append(eqn_lambda)

#%%

derivative_trajectory_model = np.array(dx_model).T
derivative_trajectory_real =  np.array(dx_real).T

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
