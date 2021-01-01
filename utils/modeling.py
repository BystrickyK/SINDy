from utils.dynamical_systems import DynamicalSystem
import numpy as np
from utils.control_structures import timeout
from utils.signal_processing import *
from utils.regression import *
import time

def simdata_to_signals(sim_data):
    # Time data
    time_data = sim_data[:, 0]

    # State data
    state_cols = np.arange(1, sim_data.shape[1] // 2 + 1)
    state_data = StateSignal(time_data, sim_data[:, state_cols])

    # Forcing data
    forcing_cols = state_cols + sim_data.shape[1] // 2
    forcing_data = ForcingSignal(time_data, sim_data[:, forcing_cols])

    # Full data
    full_data = FullSignal(state_data, forcing_data)
    return full_data


class model(DynamicalSystem):
    def __init__(self, candidate_fuctions, ksi, x0, dt=0.01, t0=0, solver='RK45'):

        self.candidate_functions = candidate_fuctions
        self.ksi = ksi

        self.fun, self.complexity, self.fun_str = self.create_SINDy_model(self.candidate_functions, self.ksi)

        self.fit = None


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



class RigorousIdentificator:
    def __init__(self, measurements, inputs, verbose=True):
        """
        Takes regressor inputs (state measurements and inputs)
        and creates many different models by varying regression hyperparameters
        Args:
            measurements:
            inputs:
            verbose:
        """
        self.measurements = measurements
        self.inputs = inputs

        self.verbose = verbose

        self.models = []

    def set_hyperparameter_space(self, thresh_lims=(0.01, 0.8), thresh_n=10, alpha_lims=(0.0, 0.5), alpha_n=3):
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

    def simulate_ground_truth(self, model, u_fun, t0=0, t1=1, t2=3, solver='RK45'):
        sys = DynamicalSystem(fun=model, x0=self.x0, dt=self.dt, t0=t0, solver=solver)
        if t1 > 0:
            sys.propagate(t1)
        if t2 > 0:
            sys.propagate_forced(t2, u_fun)

        sig = simdata_to_signals(sys.sim_data)
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
