from utils.dynamical_systems import LorenzSystem, LotkaVolterraSystem
from equations.Lorenz import lorenz_equation
from equations.LotkaVolterra import lotka_volterra_equation
from utils.function_libraries import poly_library
from utils.signal_processing import StateSignal, ForcingSignal, ProcessedSignal, FullSignal
from utils.modeling import RigorousIdentificator
from utils.model_selection import ModelSelector
from utils.visualization import *
from utils.signals import RandomWalk
import pandas as pd


# Simulate the dynamical system
dt = 0.0025
sys = LorenzSystem([18, 18, 18], dt=dt)

rand_walk1 = RandomWalk(300, dispersion=20, fs=int(1/dt), freq_cutoff=2)
rand_walk2 = RandomWalk(300, dispersion=25, fs=int(1/dt), freq_cutoff=2)
rand_walk3 = RandomWalk(300, dispersion=20, fs=int(1/dt), freq_cutoff=2)

sys.propagate(1)
u0 = lambda t, x: 0.09*(25-x[0]) + rand_walk1(t)
u1 = lambda t, x: 0.09*(25-x[1]) + rand_walk2(t)
u2 = lambda t, x: 0.09*(25-x[2]) + rand_walk3(t)
u_fun = (u0, u1, u2)
sys.propagate_forced(10, u_fun)


# Time data
time_data = sys.sim_data[:, 0]
# State data
state_data = StateSignal(time_data, sys.sim_data[:, [1, 2, 3]])
# Forcing data
forcing_data = ForcingSignal(time_data, sys.sim_data[:, [4, 5, 6]])

#%%
# Load the lorenz system function for analytical derivative computation
true_model = lorenz_equation()
# System dimensionality
dims = (sys.sim_data.shape[1]-1)//2
# Create a ProcessedSignal instance - calculate derivatives, filter out noise etc.
sig = ProcessedSignal(
    time_data, state_data.x.values, forcing_data.u.values,
    noise_power=0,
    spectral_cutoff=0,
    kernel='flattop',
    kernel_size=80,
    model=true_model
)

# %%

# Plot simulation data
# plot_tvector(sig.t, sig.x.values, 'x', title=r'$State \ \mathbf{X}$')
# plot_tvector(sig.t, sig.u.values, 'u', title=r'$Forcing \ \mathbf{U}$')
# plot_tvector(sig.t, sig.dxdt_exact.values, '\dot{x}', title=r'$State \ derivative \  \mathbf{\dot{X}}$')
plot_dxdt_comparison(sig)
plot_svd(sig.svd)
plot_lorentz3d(sig.x_filtered.values, title='Filtered training trajectory')
plot_lorentz3d(sig.x.values, title='Unfiltered training trajectory')
plt.show()
# %%
# SINDy
# dx = sig.dxdt_spectral_filtered
# x = sig.x_filtered
# dx = sig.dxdt_spectral
# x = sig.x
dx = sig.dxdt_exact
x = sig.x_clean
u = sig.u

xu = pd.concat([x, u], axis=1)
theta = poly_library(xu, poly_orders=(1, 2, 3))

dx = cutoff(dx, sig.kernel_size*2)
x = cutoff(x, sig.kernel_size*2)
theta = cutoff(theta, sig.kernel_size*2)
u = cutoff(forcing_data.u, sig.kernel_size*2)



# %%

# u0 = lambda t, x: 0.05*(25-x[0])
# u1 = lambda t, x: 0.05*(25-x[1])
# u2 = lambda t, x: 0.05*(25-x[2])

u0 = lambda t, x: rand_walk1(t)*0.1
u1 = lambda t, x: rand_walk2(t)*0.1
u2 = lambda t, x: rand_walk3(t)*0.1
u_fun = (u0, u1, u2)

rigorous = RigorousIdentificator(None, None)
rigorous.set_theta(theta)
rigorous.set_state_derivative(dx)
rigorous.set_test_params(x0=[5, -15, 10], dt=0.0025)
rigorous.set_hyperparameter_space()
rigorous.simulate_ground_truth(true_model, u_fun, t1=1, t2=3)
rigorous.create_models()

#%%
# u0 = lambda t, x: 0.05*(25-x[0])
# u1 = lambda t, x: 0.05*(25-x[1])
# u2 = lambda t, x: 0.05*(25-x[2])
# u_fun = (u0, u1, u2)

Selector = ModelSelector(rigorous.models, u_fun, truth=rigorous.ground_truth)
Selector.sort_by_complexity()
Selector.test_models(t1=1, t2=3)
#%%
models = [mdl for mdl in Selector.evaluation if mdl['success']]
rss = np.array([mdl['RSS'] for mdl in models])
bics = np.array([mdl['BIC'] for mdl in models])
complexities = np.array([mdl['complexity'] for mdl in models])
aics = np.array([mdl['AIC'] for mdl in models])
width = 0.4
#
fig, ax = plt.subplots(nrows=2, tight_layout=True, figsize=(12, 5))
ax[0].bar(np.arange(len(models))-width/2, np.log(rss), width, label='log(RSS)', color='tab:blue')
ax02 = ax[0].twinx()
ax02.bar(np.arange(len(models))+width/2, complexities, width, label='complexity', color='tab:red')
ax[0].set_xticks(np.arange(len(models)))
ax[0].set_ylabel("Log(RSS)", color='tab:blue')
ax[0].tick_params(axis='y', labelcolor='tab:blue')
ax02.tick_params(axis='y', labelcolor='tab:red')
ax02.set_ylabel("Complexity", color='tab:red')


ax[1].bar(np.arange(len(models))+width/2, aics, width, label='AIC', color='tab:blue')
ax12 = ax[1].twinx()
ax12.bar(np.arange(len(models))-width/2, bics, width, label='BIC', color='tab:red')
ax[1].set_ylabel("AIC", color='tab:blue')
ax[1].tick_params(axis='y', labelcolor='tab:blue')
ax12.tick_params(axis='y', labelcolor='tab:red')
ax12.set_ylabel("BIC", color='tab:red')
ax[1].set_xticks(np.arange(len(models)))
ax[1].set_xlabel("Model")
#%%
best_idx = np.argmin(aics)
# selection = models[best_idx]
selection = models[2]
mdl = selection['model']
cmplx = selection['complexity']
aic = selection['AIC']
bic = selection['BIC']
residuals = selection['residuals']

print(f'Best model: {best_idx}\nComplexity: {cmplx}\tBIC: {bic:.2f}\tAIC: {aic:.2f}')

plot_ksi_fig(mdl.ksi, theta, dx, title=r'$\Xi$')

axt = plot_lorentz3d(rigorous.ground_truth.x.values, title='comparison')
plot_lorentz3d_ax(mdl.sim_data.x.values, axt)
plt.legend({'Real', 'Model'})

plot_lorentz3d(residuals.values, title='residuals', color='red')
