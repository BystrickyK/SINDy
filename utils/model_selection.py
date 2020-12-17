import numpy as np
import time
from utils.dynamical_systems import DynamicalSystem
from utils.control_structures import timeout
from utils.signal_processing import StateSignal, ForcingSignal, FullSignal


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


class ModelSelector:

    def __init__(self, models, input_function, truth=None, true_model=None):

        if truth is None:
            self.true_model = true_model
            self.truth = self.simulate_testing_scenario(true_model)
            self.input_function = input_function
        else:
            self.truth = truth
            self.input_function = input_function

        self.models = np.array(models)
        self.evaluation = []

    def sort_by_complexity(self):
        mdl_complexities = [model.complexity for model in self.models]
        mdl_idxs = np.argsort(mdl_complexities)
        self.models = self.models[mdl_idxs]

    def simulate_testing_scenario(self, model, t1, t2, solver='RK45'):
        # u_fun => control law
        sys = DynamicalSystem(fun=model.fun,
                              x0=self.truth.x.values[0, :],
                              dt=self.truth.t[1] - self.truth.t[0],
                              t0=self.truth.t[0],
                              solver=solver)

        with timeout(5):
            try:
                if t1 > 0:
                    sys.propagate(t1)
                if t2 > 0:
                    sys.propagate_forced(t2, self.input_function)
                model.test_success = True
            except Exception as ex:
                print(ex)
                model.test_success = False
                return

        full_data = simdata_to_signals(sys.sim_data)
        model.sim_data = full_data
        return full_data

    def test_models(self, t1=1, t2=3):

        for i, model in enumerate(self.models):
            time_start = time.time()
            self.simulate_testing_scenario(model, t1, t2)
            if model.test_success:
                residuals = self.truth.x - model.sim_data.x
                norm_resid = residuals.apply(np.linalg.norm, axis=1)

                # https://en.wikipedia.org/wiki/Akaike_information_criterion "Comparison with least squares"
                rss = np.sum(np.square(norm_resid))  # Residual sum of squares
                n = len(residuals)
                aic = 2*model.complexity + n*np.log(rss/n)
                bic = np.log(n)*model.complexity + n*np.log(rss/n)

                self.evaluation.append({'model': model,
                                        'AIC': aic,
                                        'RSS': rss,
                                        'residuals': residuals,
                                        'success': model.test_success,
                                        'complexity': model.complexity,
                                        'BIC': bic})
                time_end = time.time()
                print(
                    "#{}\n\tAIC: {:.2f}\tRSS: {:.0f}\n\tRuntime: {:0.2f}ms\n".format(i, aic, rss, (time_end - time_start) * 10 ** 3))

            else:
                self.evaluation.append({'model': model,
                                        'success': model.test_success,
                                        'complexity': model.complexity})
                time_end = time.time()
                print(
                    "#{} (Failed)\n\tAIC: X: X\tRSS: X\tFit: X\nRuntime: {:0.2f}ms\n".format(i, (time_end - time_start) * 10 ** 3))


# errors = []
# i_ = []
# for i, mdl in enumerate(mdls_by_complexity[:, 0]):
#     time_start = time.time()
#     # print(mdl.fun_str)
#     mdl.simulate_testing_scenario(u_fun)
#     if mdl.test_success:
#         err = mdl.test_simdata.x - rigorous.ground_truth.x
#         err = err.apply(np.linalg.norm, axis=1)
#         mean_abs_dev = err.mean()
#     else:
#         mean_abs_dev = -5
#     errors.append(mean_abs_dev)
#     i_.append(i)
#     time_end = time.time()
#
#     print("#{}\n\tMAD: {}\tRuntime: {:0.2f}ms\n".format(i, mean_abs_dev, (time_end-time_start)*10**3))
