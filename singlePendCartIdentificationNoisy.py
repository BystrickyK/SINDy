import pandas as pd
import matplotlib.pyplot as plt
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.solution_processing import *
import matplotlib as mpl
import os
import pickle
import sympy as sp
from sympy.utilities.codegen import codegen
from decimal import Decimal

# fft in moving window & averaging

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata_dt001.csv'

sim_data = pd.read_csv(filename)
sim_data = pd.concat([sim_data, sim_data[::-1]]).reset_index(drop=True)
dt = sim_data.iloc[1, 0] - sim_data.iloc[0, 0]

# Generate a StateSignal object from the measurement data and add noise
rnp = 0.2*np.array([0.02, 0.0001, 0.05, 0.05])
Xm = StateSignal(sim_data.iloc[:, 1:-1], dt=dt, relative_noise_power=tuple(rnp))

# Real signals
Xc = StateSignal(Xm.values_clean, dt)  # Use clean data
DXc = StateDerivativeSignal(Xc, method='finitediff')

x = []
y = []
for i in range(30):

    # 0th order signals, positions
    Xn = StateSignal(Xm.values.iloc[:, 0:2], dt)
    filt = SpectralFilter(Xn)
    # gamma = 0.3 + i*0.08
    # filt.find_cutoffs(k=0.98, gamma=gamma, plot=False)  # Find the bandwidth of strong frequencies
    # X = filt.filter(plot=False)  # Set all weaker frequencies to 0
    # cutoff_frequencies = filt.cutoffs
    gamma = (i*4)+1
    filt = KernelFilter(kernel_size=gamma)
    X = filt.filter(Xn.values)

    # 1st order signals, velocities
    X = StateSignal(X.values, dt)
    DX = StateDerivativeSignal(X, method='spectral', kernel_size=1)
    # DXf = filt.filter(x=DX.values.values, dt=dt, var_label='dx')
    # DXf.columns = [f'dx_{i}' for i in [1, 2]]

    # 2nd order signals, accelerations
    DDX = StateDerivativeSignal(DX, method='spectral', kernel_size=1)
    DDX.values.columns = [f'dx_{i}' for i in [3, 4]]
    # DDXf = filt.filter(x=DDX.values.values, dt=dt, var_label='dx')
    # DDXf.columns = [f'dx_{i}' for i in [3, 4]]

    # Get full state vars / state var derivatives
    N = int(sim_data.shape[0]/2)

    X_full = pd.concat([X.values, DX.values], axis=1)
    X_full = X_full.iloc[:N, :]
    X_full.columns = [f'x_{i}' for i in [1,2,3,4]]

    DX_full = pd.concat([DX.values, DDX.values], axis=1)
    DX_full = DX_full.iloc[:N, :]
    DX_full.columns = [f'dx_{i}' for i in [1,2,3,4]]

    # Gather the signals for easy plotting
    X = X.values
    X_c = Xc.values.iloc[:, 0:2]
    X_n = Xm.values.iloc[:, 0:2]
    DX = DX.values
    # DX = DXf
    DX_c = Xc.values.iloc[:, 2:4]
    # DDX = DDXf
    DDX = DDX.values
    DDX_c = DXc.values.iloc[:, 2:4]

    # fig, axs = plt.subplots(nrows=X.shape[1], ncols=3, sharex=True, tight_layout=True)
    # left = N//2
    # right = N
    # for i in range(axs.shape[0]):
    #     axs[i,0].plot(X.iloc[left:right, i], linewidth=1)
    #     axs[i,0].plot(X_c.iloc[left:right, i], alpha=0.7)
    #     axs[i,0].plot(X_n.iloc[left:right, i], alpha=0.4, linewidth=0.5)
    #     axs[i,0].legend(["Spectral-Filtered", "Clean", "Noise"])
    #     axs[i,1].plot(DX.iloc[left:right, i], linewidth=1)
    #     axs[i,1].plot(DX_c.iloc[left:right, i], alpha=0.7)
    #     axs[i,1].legend(["Spectral-Diff", "Clean"])
    #     axs[i,2].plot(DDX.iloc[left:right, i], linewidth=1)
    #     axs[i,2].plot(DDX_c.iloc[left:right, i], alpha=0.7)
    #     axs[i,2].legend(["Spectral-DoubleDiff", "Clean"])
    # plt.show()

    ddx_error = np.mean(np.sqrt(np.square(DX_full-DXc.values)))
    print(f"Gamma: {gamma}\nMean error:\n {ddx_error}\n")
    x.append(gamma)
    y.append(ddx_error[3])
plt.figure()
plt.semilogy(x, y, '-o')
plt.show()

#%%
# state_data = X.x
N = X_full.values.shape[0]


u = ForcingSignal(sim_data.iloc[:, -1], dt)


state_data = X_full.iloc[:int(N/2), :].reset_index(drop=True)
state_derivative_data = DX_full.iloc[:int(N/2), :].reset_index(drop=True)
input_data = u.values.iloc[:int(N/2), :].reset_index(drop=True)

dim = state_data.shape[1]
# %%
# Build library with sums of angles (state var 2) and its sines/cosines
trig_data = trigonometric_library(state_data.iloc[:, 1:dim // 2])
trig_data = poly_library(trig_data, (1, 2))

v_sq = square_library(state_data.iloc[:, 3:4])

trig_v_sq = product_library(trig_data, v_sq)
trig_v_sq = trig_v_sq.iloc[:, 1:]

trig_bilinears = product_library(trig_data, pd.concat((state_data.loc[:, ('x_3', 'x_4')],
                                                       state_derivative_data.loc[:, ('dx_3', 'dx_4')],
                                                       input_data), axis=1))

bad_idx = np.array(['1*' in colname or 'sin(x_2)*sin(x_2)' in colname for colname in trig_bilinears.columns])
trig_bilinears = trig_bilinears.iloc[:, ~bad_idx]
# linear/angular accelerations -> second half of state var derivatives
# %%
theta = pd.concat([state_data, state_derivative_data, input_data, trig_data, trig_v_sq, trig_bilinears], axis=1)
theta = theta.loc[:, ~theta.columns.duplicated()]

dump_idx = np.array([col.count('sin') == 2 for col in theta.columns])  # Find indices of cols that contain sin^2(x)
theta = theta.iloc[:, ~dump_idx]  # Keep all other columns

# Plot the correlation matrix of the regression matrix
corr = theta.corr()
plot_corr(corr, theta.columns, labels=False)
plt.show()

# %% Compute the solution or retrieve it from cache

rewrite = True # Should the cache be rewritten
eqns_to_identify = ['dx_3', 'dx_4']  # State derivatives whose equation we want to identify
cache_str = 'singlePendSolutionsNoisyDoubleDiff2'
eqns_models = {}
for eqn in eqns_to_identify:
    # find cols with other state derivatives than the one currently being identified
    idx = np.array([('d' in col and eqn not in col) for col in theta.columns])
    # Construct a library for identifying the desired equation
    theta_hat = theta.loc[:, ~idx]  # and keep all cols except them

    eqns_models[eqn] = {}
    eqns_models[eqn]['theta'] = theta_hat

    cachename = dirname + cache_str + '_' + eqn

    if os.path.exists(cachename) and not rewrite:
        print("Retrieving solution from cache.")
        with open(cachename, 'rb') as f:
            eqns_models[eqn] = pickle.load(f)
    else:
        print("No solution in cache, calculating solution from scratch.")
        EqnIdentifier = PI_Identifier(theta_hat)
        EqnIdentifier.set_thresh_range(lims=(0.00001, 0.1), n=10)
        EqnIdentifier.create_models(n_models=theta_hat.shape[1], iters=7, shuffle=False)
        eqns_models[eqn]['models'] = EqnIdentifier.all_models
        with open(cachename, 'wb') as f:
            pickle.dump(eqns_models[eqn], f)
# %%
dynamic_model = {}
for eqn_str, eqn_model in eqns_models.items():
    theta = eqn_model['theta']
    models = eqn_model['models']
    dynamic_model[eqn_str] = {}

    # %% Remove duplicate models
    models = unique_models(models, theta.columns)
    models = models.loc[models.fit > 0.7, :].reset_index(drop=True)

    eqns = [[*zip(np.round(model.sol[model.active], 3), theta.columns[model.active])]
            for idx, model in models.iterrows()]
    eqns = [[str(par) + '*' + term for par, term in eqn] for eqn in eqns]
    eqns = [' + '.join(eqn) for eqn in eqns]

    idx = ['d' in eqn for eqn in eqns]
    models = models.loc[idx, :].reset_index()

    idx = []
    # %% Visualize the solutions -> calculate and plot activation distance matrix
    # and plot the matrix of implicit solutions
    dist = distance_matrix(models, plot=False)

    # %% Look for consistent models by finding clusters in the term activation space
    models = consistent_models(models, dist,
                               min_cluster_size=2)

    plot_implicit_sols(models, theta.columns, show_labels=True)
    plt.show()

    # %% Decompose one of the models
    # choice = int(input("Choose model index:"))
    choice = models.fit.argmax()
    model = models.iloc[choice, :]

    fit = model['fit']
    active_terms = theta.iloc[:, model['active']].values
    term_labels = theta.columns[model['active']]
    parameters = np.array(model['sol'])[model['active']]
    signals = parameters * active_terms
    solution_string = ' + \n'.join(
        ['$' + str(round(par, 3)) + '\;' + term + '$' for par, term in zip(parameters, term_labels)]) + '\n$= 0$'

    dynamic_model[eqn_str]['fit'] = fit

    fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    ax = np.reshape(ax, [-1, ])
    ax[0].plot(signals)
    ax[0].legend(term_labels, borderpad=0.5, frameon=True, fancybox=True, framealpha=0.7)
    ax[0].set_title(f'Fit: {round(fit, 4)}\nModel terms')

    residuals_sq = np.sum(np.square(signals), axis=1)
    ax[1].plot(residuals_sq)
    ax[1].set_title(rf'Sum of squares of residuals: {Decimal(np.sum(residuals_sq)):.3E}' +
                    '\nSquares of residuals')

    term_energy = np.sum(np.square(signals), axis=0)
    ax[2].bar(range(len(parameters)), term_energy,
              tick_label=term_labels)
    ax[2].set_title(rf'Term energies')
    ax[2].xaxis.set_tick_params(rotation=90)

    ax[3].grid(False)
    ax[3].set_xticklabels([])
    ax[3].set_yticklabels([])
    ax[3].text(0.2, -0.2, rf'{solution_string}', fontsize=12)

    plt.show()

    # %%
    # Construct implicit function string
    eqn = [*zip(np.round(model.sol[model.active], 3), theta.columns[model.active])]
    eqn = [str(par) + '*' + term for par, term in eqn]
    eqn = ' + '.join(eqn)
    # Parse the string into a sympy expression
    symeqn = sp.parse_expr(eqn)
    symeqn = sp.solve(symeqn, eqn_str)[0]
    # Lambdify the sympy expression for evaluation
    vars = ['x_1', 'x_2', 'x_3', 'x_4', 'u']
    vars = [sp.parse_expr(var) for var in vars]
    lameqn = sp.lambdify(vars, symeqn)
    eqn_lambda = lambda row: lameqn(*row)  # one input in the form [x1, x2, ..., xn, u]
    dynamic_model[eqn_str]['lmbda'] = eqn_lambda
    dynamic_model[eqn_str]['symeqn'] = symeqn
    dynamic_model[eqn_str]['str'] = symeqn

    data = pd.concat([state_data, input_data], axis=1)
    eqn_data = data.apply(eqn_lambda, axis=1)

    print(f'Eqn: {eqn_str}\nEquation fit: {fit}\n')

symeqns = [dynamic_model[eqn]['symeqn'] for eqn in eqns_to_identify]
codegen(('identified_model2', symeqns),
        language='octave', to_files=True)
