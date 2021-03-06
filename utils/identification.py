from utils.dynamical_systems import DynamicalSystem
import numpy as np
from utils.control_structures import timeout
from utils.modeling import simdata_to_signals
from utils.regression import *
from collections import namedtuple
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

class PI_Identificator:
    def __init__(self, theta, verbose=True):
        """
        Takes in the regressor matrix and identifies potential state potential
        models by guessing active regressors and varying regression hyperparameters
        Args:
            measurements:
            inputs:
            verbose:
        """
        self.theta = theta
        self.results = []

        self.verbose = verbose

    def set_thresh_range(self, lims=(0.01, 0.8), n=10):
        thresh = np.linspace(lims[0], lims[1], n, endpoint=True)
        self.thresh = thresh

    def set_theta(self, theta):
        self.theta = theta

    def create_models(self, n_models=10, iters=10):

        # Set of possible LHS functions (defined as column indices from function library)
        lhs_candidate_inds = set([*range(n_models)])

        # Define named tuple form
        IdentfiedEqn = namedtuple('Eqn', ['lhs', 'rhs', 'complexity'])

        self.n_models = []
        for n in range(n_models):

            n_start = time.time()
            j = np.random.choice(list(lhs_candidate_inds))    # Choose a random column index
            lhs_candidate_inds.remove(j)  # Remove the chosen index from the set, so it isn't chosen again in next iterations

            j_models = []
            for i, hyperparameter in enumerate(self.thresh, start=1):

                i_start = time.time()
                lhs = self.theta.iloc[:, j]  # Choose j-th column of theta as LHS guess

                if self.verbose:
                    print(f'#{n}\nLHS guess:\t\t{lhs.name}\n')

                # Find sparse solution using STLQ
                rhs, valid = seq_thresh_ls(A=self.theta, b=lhs, n=iters, threshold=hyperparameter)

                eqn = IdentfiedEqn(lhs, rhs, np.linalg.norm(rhs, 0))
                i_end = time.time()
                j_models.append(eqn)

                if self.verbose:
                    print(f'Runtime:\t\t{i_end}\nComplexity:\t\t{eqn.complexity}\n')

            n_end = time.time()

            sparsest_sol = np.argmin([model.complexity for model in j_models])
            sol = j_models[sparsest_sol]
            self.n_models.append(sol)

            if self.verbose:
                print("Created model #{i} / {total}\n\t".format(i=n, total=len(self.thresh)) +
                      "Model complexity: {cmplx}\n\t".format(cmplx=sol.complexity) +
                      "Iteration runtime: {:0.2f}ms\n".format((n_end-n_start)*10**3))
