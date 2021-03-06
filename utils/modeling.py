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
