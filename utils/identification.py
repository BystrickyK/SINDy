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

class PI_Identifier:
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

    def create_models(self, n_models=10, iters=10, shuffle=True):

        # Set of possible LHS functions (defined as column indices from function library)
        lhs_candidate_inds = set([*range(len(self.theta.columns))])

        # Define named tuple form
        IdentfiedEqn = namedtuple('Eqn', ['lhs_str', 'rhs_str', 'rhs_sol', 'complexity', 'residuals'])

        self.n_models = []
        self.all_models = []
        for n in range(n_models):

            n_start = time.time()
            if shuffle:
                j = np.random.choice(list(lhs_candidate_inds))    # Choose a random column index
            else:
                j = n
            lhs_candidate_inds.remove(j)  # Remove the chosen index from the set, so it isn't chosen again in next iterations
            print('\n\n')

            j_models = []
            for i, hyperparameter in enumerate(self.thresh, start=1):

                i_start = time.time()
                lhs = self.theta.iloc[:, j]  # Choose j-th column of theta as LHS guess
                theta = self.theta.drop(lhs.name, axis=1)
                if self.verbose:
                    print(f'\n#{n+1}\nLHS guess:\t\t{lhs.name}')

                # Find sparse solution using STLQ
                rhs, valid, residuals = seq_thresh_ls(A=theta, b=lhs, n=iters, alpha=0, threshold=hyperparameter)

                nnz_idx = np.nonzero(rhs)[0]
                rhs_str = ["{:3f}".format(rhs[i]) + '*' + theta.columns[i] for i in nnz_idx]
                rhs_str = " + ".join(rhs_str)

                eqn = IdentfiedEqn(lhs.name, theta.columns, rhs, np.linalg.norm(rhs, 0), residuals)
                i_end = time.time()
                j_models.append(eqn)

                if self.verbose:
                    print('Runtime:\t\t{:0.2f}ms\nComplexity:\t\t{}\nRHS:\t\t{}\nSSQ:\t\t{}'.format(
                        (i_end-i_start)*10**3, eqn.complexity, rhs_str, residuals))

            n_end = time.time()

            complexities = np.array([model.complexity for model in j_models])
            residuals = np.array([model.residuals for model in j_models])
            nnz_models = np.greater(complexities, 0)  # nonzero models
            nondense_models = np.less(complexities, theta.shape[1]-2)
            viable_models = np.all([nnz_models, nondense_models], axis=0)
            sol = j_models[np.argmin(residuals)]
            # if viable_models.sum() > 0:
            #     best_sol = np.argmin(residuals[viable_models])
            #     sol = j_models[viable_models][best_sol]
            # else:
            #     sol = j_models[0]
            self.n_models.append(sol)
            self.all_models.extend(list(np.array(j_models)[viable_models]))

            if self.verbose:
                rhs = sol.rhs_sol
                lhs_str = sol.lhs_str

                nnz_idx = np.nonzero(rhs)[0]
                rhs_str = ["{:3f}".format(rhs[i]) + '*' + theta.columns[i] for i in nnz_idx]
                rhs_str = " + ".join(rhs_str)


                run_info = ("Created model #{i} / {total}\n\t".format(i=n+1, total=n_models) +
                      "Model complexity: {cmplx}\n\t".format(cmplx=sol.complexity) +
                      "Iteration runtime: {:0.2f}ms\n".format((n_end-n_start)*10**3))

                result_str = ("LHS: {}\n".format(lhs_str) +
                      "RHS: {}\n".format(rhs_str))

                print(run_info + result_str)

                file = open("results.txt", "a")
                file.write(result_str + '\n')
                file.close()

                dx_strs = ['dx[' + str(i) + ']' for i in [5,6]]
                potential_sol = [((dxstr in rhs_str) or (dxstr in lhs_str)) for dxstr in dx_strs]
                potential_sol = np.any(potential_sol)
                if potential_sol:
                    file = open("results_dx.txt", "a")
                    file.write(result_str + '\n')
                    file.close()
