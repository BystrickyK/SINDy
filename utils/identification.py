from utils.dynamical_systems import DynamicalSystem
import numpy as np
from utils.control_structures import timeout
from utils.modeling import simdata_to_signals
from utils.regression import *
from utils.model_selection import calculate_fit
from collections import namedtuple
# from dask import delayed
import dask
import time

class Term:
    def __init__(self, var, param):
        self.var = var
        self.param = param

    def __str__(self):
        print(f'{self.param}{self.var}')

class Equation:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class PI_Identifier:
    def __init__(self, theta_train, theta_val, verbose=False):
        """
        Takes in the regressor matrix and identifies potential state potential
        models by guessing active regressors and varying regression hyperparameters
        Args:
            theta: Function library matrix
        """
        self.theta_train = theta_train
        self.theta_val = theta_val
        self.results = []

        self.verbose = verbose

        # Define named tuple form
        self.IdentfiedEqn = namedtuple('Eqn', ['lhs_str', 'rhs_str',
                                          'rhs_sol', 'complexity',
                                          'residuals', 'trainerror', 'valerror'])

        print("Created new identifier object.\n")

    def set_thresh_range(self, lims=(0.01, 0.8), n=10):
        thresh = np.linspace(lims[0], lims[1], n, endpoint=True)
        self.thresh = thresh

    def create_model(self, n, guess_index, hyperparameter, iters=10):
        i_start = time.time()
        lhs_train = self.theta_train.iloc[:, guess_index]  # Choose j-th column of theta as LHS guess
        theta_train = self.theta_train.drop(lhs_train.name, axis=1)
        lhs_val = self.theta_val.iloc[:, guess_index]  # Choose j-th column of theta as LHS guess
        theta_val = self.theta_val.drop(lhs_val.name, axis=1)
        if self.verbose:
            print(f'\n#{n+1}\nLHS guess:\t\t{lhs_train.name}')

        # Find sparse solution using STLQ
        # rhs, valid, residuals = seq_thresh_ls(A=theta, b=lhs, n=iters, threshold=hyperparameter)
        # rhs, valid, residuals = seq_energy_thresh_ls(A_train=theta_train, b_train=lhs_train,
        #                                                  n=iters, lambda_=hyperparameter, verbose=False)

        rhs, valid, residuals = seq_energy_thresh_ls_val(A_train=theta_train, b_train=lhs_train,
                                                         A_val=theta_val, b_val=lhs_val,
                                                         n=iters, lambda_=hyperparameter, verbose=False)


        fit_train, error_train = calculate_fit(theta_train.values, rhs, lhs_train)
        fit_val, error_val = calculate_fit(theta_val.values, rhs, lhs_val)

        eqn = self.IdentfiedEqn(lhs_train.name, theta_train.columns,
                           rhs, np.linalg.norm(rhs, 0),
                           residuals, error_train, error_val)
        i_end = time.time()

        if self.verbose:
            nnz_idx = np.nonzero(rhs)[0]
            rhs_str = ["{:3f}".format(rhs[i]) + '*' + theta_train.columns[i] for i in nnz_idx]
            rhs_str = " + ".join(rhs_str)
            print('Runtime:\t\t{:0.2f}ms\nComplexity:\t\t{}\nRHS:\t\t{}\nFit:\t\t{}'.format(
                (i_end-i_start)*10**3, eqn.complexity, rhs_str, fit_train))

        return eqn

    # @dask.delayed
    def create_models_for_lhs_guess(self, lhs_guess_term_index, iters, n, n_models):

        j_models = []
        for i, hyperparameter in enumerate(self.thresh, start=1):
            eqn = self.create_model(n, lhs_guess_term_index, hyperparameter, iters=iters)
            j_models.append(eqn)

        complexities = np.array([model.complexity for model in j_models])
        nnz_models_idx = np.greater(complexities, 0)  # nonzero models

        nnz_models = [*np.array(j_models, dtype=object)[nnz_models_idx]]
        if len(nnz_models)==0:
            return nnz_models

        errors_val = np.array([model[6] for model in nnz_models])
        errors_train = np.array([model[5] for model in nnz_models])

        run_info = ("Created models #{i} / {total}\n\tFor LHS guess: {lhs}\n\tBest train error: {trainerror}\n\tBest val error: {valerror}\n".format(
            i=n + 1,
            total=n_models,
            lhs=self.theta_train.columns[lhs_guess_term_index],
            trainerror=round(errors_train.min(), 3),
            valerror=round(errors_val.min(), 3)))

        print(run_info)

        return nnz_models

    def create_models(self, n_models=10, iters=10, shuffle=True):
        t_start = time.time()
        # Set of possible LHS functions (defined as column indices from function library)
        lhs_candidate_inds = set([*range(len(self.theta_train.columns))])

        lhs_models = []
        self.all_models = []
        for n in range(n_models):

            if shuffle:
                 lhs_guess_term_index = np.random.choice(list(lhs_candidate_inds))    # Choose a random column index
            else:
                lhs_guess_term_index = n
            lhs_candidate_inds.remove(lhs_guess_term_index)  # Remove the chosen index from the set, so it isn't chosen again in next iterations

            nnz_models = self.create_models_for_lhs_guess(lhs_guess_term_index, iters, n, n_models)
            lhs_models.append(nnz_models)

            # self.all_models.extend(list(np.array(j_models, dtype=object)[nnz_models]),)
        # lhs_models = dask.delayed(lhs_models).compute()
        lhs_models = [models for models in lhs_models if len(models)>0]
        self.all_models = np.vstack(lhs_models)
        t_end = time.time()
        print(f"Total runtime: {round(t_end-t_start, 2)} sec")