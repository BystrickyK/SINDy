from dynamical_systems import LorenzSystem, DynamicalSystem, LotkaVolterraSystem
from equations.Lorenz import lorenz_equation
from equations.LotkaVolterra import lotka_volterra_equation
from function_library_creators import poly_library
from signal_processing import StateSignal, ForcingSignal, ProcessedSignal
import time as time
from utils.regression import seq_thresh_ls
from utils.visualization import *
from utils.signals import RandomWalk
from utils.control_structures import timeout
import pandas as pd


# Simulate the dynamical system
dt = 0.0025
sys = LorenzSystem([18, 18, 18], dt=0.0025)
# sys = LotkaVolterraSystem([5, 10], dt=0.0025)
rand_walk1 = RandomWalk(300, dispersion=5, fs=int(1/dt), freq_cutoff=0.3)
rand_walk2 = RandomWalk(300, dispersion=5, fs=int(1/dt), freq_cutoff=0.3)
rand_walk3 = RandomWalk(300, dispersion=5, fs=int(1/dt), freq_cutoff=0.3)

sys.propagate(1)
u0 = lambda t, x: 0.2*(26-x[0]) + rand_walk1(t)
u1 = lambda t, x: 0.2*(25-x[1]) + rand_walk2(t)
u2 = lambda t, x: 0.2*(20-x[2]) + rand_walk3(t)
u_fun = (u0, u1, u2)
sys.propagate_forced(30, u_fun)


# Time data
time_data = sys.sim_data[:, 0]
# State data
state_data = StateSignal(time_data, sys.sim_data[:, [1, 2, 3]])
# Forcing data
forcing_data = ForcingSignal(time_data, sys.sim_data[:, [4, 5, 6]])

# Load the lorenz system function for analytical derivative computation
true_model = lorenz_equation()
# model = lotka_volterra_equation()
# System dimensionality
dims = (sys.sim_data.shape[1]-1)//2
# Create a ProcessedSignal instance - calculate derivatives, filter out noise etc.
sig = ProcessedSignal(
    time_data, state_data.x.values, forcing_data.u.values,
    noise_power=0.1,
    spectral_cutoff=0.1,
    kernel='flattop',
    kernel_size=64,
    model=true_model
)

# %%

# Plot simulation data
# plot_tvector(sig.t, sig.x.values, 'x', title=r'$State \ \mathbf{X}$')
# plot_tvector(sig.t, sig.u.values, 'u', title=r'$Forcing \ \mathbf{U}$')
# plot_tvector(sig.t, sig.dxdt_exact.values, '\dot{x}', title=r'$State \ derivative \  \mathbf{\dot{X}}$')
# plot_dxdt_comparison(sig)
# plot_svd(sig.svd)
plot_lorentz3d(sig.x_clean.values)

# %%
# SINDy
# dx = sig.dxdt_spectral_filtered
# x = sig.x_filtered
# dx = sig.dxdt_spectral
x = sig.x
dx = sig.dxdt_exact
x = sig.x_clean
u = sig.u
xu = pd.concat([x, u], axis=1)
theta = poly_library(xu, poly_orders=(1, 2, 3))

# dx = cutoff(dx, sig.kernel_size)
# x = cutoff(x, sig.kernel_size)
# theta = cutoff(theta, sig.kernel_size)
# u = cutoff(forcing_data.u, sig.kernel_size)

# theta = pd.concat([theta, u], axis=1)

# ksi = seq_thresh_ls(theta, dx, n=70, alpha=0.03, verbose=False, threshold=0.05)
# KSI = []
# KSI_thresh = []
# for thresh in np.linspace(0.01, 0.2, 30):
#     ksi = seq_thresh_ls(theta, dx, n=5, alpha=0.1, verbose=True, threshold=thresh)
#     KSI.append(ksi)
#     KSI_thresh.append(thresh)
#
# KSI = np.array(KSI)
# KSI_thresh = np.array(KSI_thresh)

# plt.bar(np.arange(0, len(KSI_complex)), KSI_complex)
# fig, ax = plt.subplots(1, 2, tight_layout=True)
# plot_ksi(ksi, theta, dx, ax[0], show_sparse=True)
# plot_ksi(ksi2, theta2, dx, ax[1], show_sparse=True)
# compare_ksi(ksi, theta, ksi2, theta2, dx)
# plot_ksi_fig(ksi, theta, dx, title=r'$\Xi$')
# plot_ksi_fig(ksi, theta, dx, title=r'$\Xi$')

# %%
class model(DynamicalSystem):
    def __init__(self, candidate_fuctions, ksi, x0, dt=0.01, t0=0, solver='RK45'):

        self.candidate_functions = candidate_fuctions
        self.ksi = ksi

        self.fun, self.complexity, self.fun_str = self.create_SINDy_model(self.candidate_functions, self.ksi)


        DynamicalSystem.__init__(self, self.fun, x0=x0, dt=dt, t0=t0, solver=solver)

        self.info = {}

    def create_SINDy_model(self, candidate_functions, ksi, thresh=0.01):
        ksiT = ksi.T
        complexity = 0  # How many parameters are nonzero

        # Full system function string
        system_str = ''
        # Build function strings for each state function out of xi coefficients and candidate function labels
        state_fun_strings = []
        for state_fun_idx in range(ksiT.shape[1]):
            system_str += "State function x{}_dot\n".format(state_fun_idx)
            state_fun_str = ''
            for cand_fun_str, cand_fun_coeff in zip(candidate_functions, ksiT[:, state_fun_idx]):
                if np.abs(cand_fun_coeff) > thresh:
                    complexity += 1
                    cand_str = "{c:0.5f} * {fun} + ".format(c=cand_fun_coeff,
                                                            fun=cand_fun_str)  # rounds to 5 decimal places
                    state_fun_str += cand_str
                    system_str += "\t{}\n".format(cand_str)
            state_fun_str = state_fun_str[:-3]  # cut off last 3 characters (the plus sign and two spaces)
            state_fun_strings.append(state_fun_str)
            system_str = system_str[:-3] + '\n\n'

        # Combine the state function strings into lambda output form
        lambda_str = 'lambda t, x, u: ['
        for state_fun_str in state_fun_strings:
            lambda_str += state_fun_str + ', '
        lambda_str = lambda_str[:-2] + ']'  # cut off last two characters and add ']'

        identified_model = eval(lambda_str)  # SINDYc identified model
        return identified_model, complexity, system_str

    def simulate_testing_scenario(self, u_fun):
         # u_fun = control law
        sys = DynamicalSystem(fun=self.fun, x0=self.x0, dt=self.dt, t0=self.t0, solver=self.solver)

        with timeout(2):
            try:
                sys.propagate(1)
                sys.propagate_forced(3, u_fun)
                self.test_success = True
            except Exception as ex:
                print(ex)
                self.test_success = False

        # Time data
        time_data = sys.sim_data[:, 0]
        # State data
        state_data = StateSignal(time_data, sys.sim_data[:, [1, 2, 3]])
        # Forcing data
        forcing_data = ForcingSignal(time_data, sys.sim_data[:, [4, 5, 6]])

        # Create a ProcessedSignal instance - just to put everything together
        sig = ProcessedSignal(
            time_data, state_data.x.values, forcing_data.u.values,
            model=self.fun
        )

        self.test_simdata = sig

    # def calculate_RMS_error(self, ground_truth):
    #     RMS_state


class RigorousIdentificator:
    def __init__(self, measurements, inputs, verbose=True):
        self.measurements = measurements
        self.inputs = inputs

        self.verbose = verbose

        self.models = []

    def set_hyperparameter_space(self, thresh_lims=(0.01, 0.5), thresh_n=10, alpha_lims=(0.0, 0.5), alpha_n=5):
        thresh = np.linspace(thresh_lims[0], thresh_lims[1], thresh_n, endpoint=True)
        alpha = np.linspace(alpha_lims[0], alpha_lims[1], alpha_n, endpoint=True)

        thresh_alpha = np.meshgrid(thresh, alpha)
        self.thresh_alpha = np.array(thresh_alpha).T.reshape(-1, 2)

    def set_theta(self, theta):
        self.theta = theta

    def set_state_derivative(self, xdot):
        self.xdot = xdot

    def set_test_params(self, x0, dt=0.01, t0=0, solver='RK45'):
        self.x0 = x0
        self.dt = dt
        self.t0 = t0
        self.solver = solver

    def simulate_ground_truth(self, model, u_fun, t0=0, solver='RK45'):
        sys = DynamicalSystem(fun=model, x0=self.x0, dt=self.dt, t0=t0, solver=solver)
        sys.propagate(1)
        sys.propagate_forced(3, u_fun)

        # Time data
        time_data = sys.sim_data[:, 0]
        # State data
        state_data = StateSignal(time_data, sys.sim_data[:, [1, 2, 3]])
        # Forcing data
        forcing_data = ForcingSignal(time_data, sys.sim_data[:, [4, 5, 6]])

        # Create a ProcessedSignal instance - just to put everything together
        sig = ProcessedSignal(
            time_data, state_data.x.values, forcing_data.u.values,
            model=model
        )

        self.ground_truth = sig

    def create_models(self, iters=10):
        for i, hyperparameter in enumerate(self.thresh_alpha, start=1):
            t_start = time.time()

            ksi, valid = seq_thresh_ls(self.theta, self.xdot, n=iters, alpha=hyperparameter[1], threshold=hyperparameter[0])
            mdl = model(self.theta.columns, ksi, x0=self.x0, dt=self.dt, t0=self.t0, solver=self.solver)
            mdl.info['hyperparameters'] = hyperparameter
            mdl.info['valid'] = valid
            self.models.append(mdl)

            t_end = time.time()
            if self.verbose:
               print("Created model #{i} / {total}\n\t".format(i=i, total=len(self.thresh_alpha)) +
                     "Threshold: {thr}\tAlpha: {alpha}\n\t".format(thr=hyperparameter[0], alpha=hyperparameter[1]) +
                     "Model complexity: {cmplx}\n\t".format(cmplx=mdl.complexity) +
                     "Iteration runtime: {:0.2f}ms\n".format((t_end-t_start)*10**3))






rigorous = RigorousIdentificator(None, None)
rigorous.set_theta(theta)
rigorous.set_state_derivative(dx)
rigorous.set_test_params(x0=[12, 12, 12], dt=0.0025)
rigorous.set_hyperparameter_space()
rigorous.simulate_ground_truth(true_model, u_fun)
rigorous.create_models()

#%%
models = []
complex = []
for model in rigorous.models:
    models.append(model)
    complex.append(model.complexity)

mdl_info = zip(models, complex)
mdls_by_complexity = np.array(sorted(mdl_info, key=lambda mdl: mdl[1], reverse=True))


plt.figure(tight_layout=True)
plt.bar(np.arange(len(mdls_by_complexity)), mdls_by_complexity[:, 1])
# id_model, info = create_SINDy_model(theta, ksi)
# print(info)


# mdl = mdls_by_complexity[35][0]
errors = []
i_ = []
for i, mdl in enumerate(mdls_by_complexity[:, 0]):
    time_start = time.time()
    # print(mdl.fun_str)
    mdl.simulate_testing_scenario(u_fun)
    if mdl.test_success:
        err = mdl.test_simdata.x - rigorous.ground_truth.x
        err = err.apply(np.linalg.norm, axis=1)
        mean_abs_dev = err.mean()
    else:
        mean_abs_dev = -5
    errors.append(mean_abs_dev)
    i_.append(i)
    time_end = time.time()

    print("#{}\n\tMAD: {}\tRuntime: {:0.2f}ms\n".format(i, mean_abs_dev, (time_end-time_start)*10**3))

#%%
model_idxs = np.arange(len(mdls_by_complexity))
complexity = mdls_by_complexity[:, 1]
width = 0.35

fig, ax = plt.subplots(tight_layout=True)

ax.bar(model_idxs - width/2, errors, width)
ax.bar(model_idxs + width/2, complexity, width)

#%%
mdl = mdls_by_complexity[8, 0]
print(mdl.fun_str)

mdl.test_simdata.x.plot(subplots=True, title="test")
rigorous.ground_truth.x.plot(subplots=True, title="truth")

plot_ksi_fig(mdl.ksi, theta, dx, title=r'$\Xi$')

