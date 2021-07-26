import numpy as np
from utils.control_structures import timeout
from utils.regression import *
from utils.model_selection import *
from collections import namedtuple
import pandas as pd
# from dask import delayed
import time

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

        eqn = {'lhs_str': [],
               'xi': [],
               'cond': [],
               'train_metric': [],
               'val_metric': []}
        # Initialize empty dataframe
        self.models = pd.DataFrame(eqn)

        self.target = None
        self.target_idx = None
        self.weights = None
        self.thresh = None

        self.guess_idxs = None
        print("Created new identifier object.\n")

    def set_thresh_range(self, lims=(0.01, 0.8), n=10):
        thresh = np.linspace(lims[0], lims[1], n, endpoint=True)
        self.thresh = thresh

    def set_weights(self, weights):
        self.weights = weights

    def set_target(self, target):
        self.target_idx = np.where(self.theta_train.columns == target)[0][0]
        self.target = target

    def set_guess_cols(self, guess_cols):
        if not hasattr(guess_cols, '__iter__'):
            guess_cols = [guess_cols]
        self.guess_idxs = guess_cols

    def create_model(self, guess_index, hyperparameter):
        """
        Given a guess index and hyperparameters, runs the regression and returns the resulting model.
        """
        # Creates a single model
        i_start = time.time()

        lhs_train = self.theta_train.iloc[:, guess_index]  # Choose j-th column of theta as LHS guess
        theta_train = self.theta_train.drop(lhs_train.name, axis=1)

        # Find sparse solution using STLQ
        # rhs, residuals, valid, cond = sequentially_thresholded_least_squares(A=theta_train, b=lhs_train,
        #                                                                      n=self.iters, lambda_=hyperparameter)
        # rhs, valid, residuals = seq_energy_thresh_ls(A_train=theta_train, b_train=lhs_train,
        #                                                  n=iters, lambda_=hyperparameter, verbose=False)

        rhs, valid, cond = sequentially_energy_thresholded_least_squares(A=theta_train, b=lhs_train,
                                                                         weights=None, target_str=self.target, n=self.iters,
                                                                         lambda_=hyperparameter, verbose=False)

        # Add the guess term parameter into the solution
        rhs = np.insert(rhs, guess_index, -1)

        # Normalize the solution so the target function coefficient is 1
        if self.target is not None:
            rhs = rhs / rhs[self.target_idx]
        else:
            # Normalize the solution so its L2 norm is 100
            rhs = rhs / np.linalg.norm(rhs)

        train_metric = calculate_rmse(self.theta_train.values, rhs, 0)
        val_metric = calculate_rmse(self.theta_val.values, rhs, 0)
        mdl = {'lhs_str': lhs_train.name,
               'xi': rhs,
               'complexity': np.sum(np.nonzero(rhs)),
               'cond': cond,
               'train_metric': train_metric,
               'val_metric': val_metric}
        self.models = self.models.append(mdl, ignore_index=True)

        i_end = time.time()

        # if self.verbose:
        #     nnz_idx = np.nonzero(rhs)[0]
        #     rhs_str = ["{:3f}".format(rhs[i]) + '*' + theta_train.columns[i] for i in nnz_idx]
        #     rhs_str = " + ".join(rhs_str)
        #     print('Runtime:\t\t{:0.2f}ms\nComplexity:\t\t{}\nRHS:\t\t{}\nFit:\t\t{}'.format(
        #         (i_end-i_start)*10**3, mdl['complexity'], rhs_str, val_metric))

        return mdl

    def create_models_for_lhs_guess(self, lhs_guess_term_index):
        """
        Given the guess column index, creates a model for each set of hyperparameters. The number of iterations
        depends on the length of self.thresh
        """

        lowest_error = 1e9
        for i, hyperparameter in enumerate(self.thresh, start=1):
            mdl = self.create_model(lhs_guess_term_index, hyperparameter)
            # print(f"{mdl['val_metric']}, {mdl['train_metric']}")
            if mdl['val_metric'] < lowest_error:
                lowest_error = mdl['train_metric']

        run_info = ("Created models for #{idx} / {total}\n\tFor LHS guess: {lhs}\n\tLowest error: {error}".format(
            idx=lhs_guess_term_index,
            total=self.n_models,
            lhs=self.theta_train.columns[lhs_guess_term_index],
            error=round(lowest_error, 3)))

        print(run_info)

    def create_models(self, n_models=None, iters=8, shuffle=True):
        """
        n_models: How many guesses should be made (if not specified, a guess is made for *each* column)
        iters: How many iterations should the SQLS algorithm do.
        shuffle: Shuffle the set of guess candidate functions before drawing a guess.
        """
        t_start = time.time()

        self.iters = iters
        self.n_models = n_models

        # If not specified, a guess is made for each column
        if n_models is None:
            self.n_models = self.theta_train.shape[1]

        # Set of possible LHS functions (defined as column indices from function library)
        if self.guess_idxs is None:
            lhs_candidate_inds = set([*range(len(self.theta_train.columns))])
        elif isinstance(self.guess_idxs, int):
            lhs_candidate_inds = set([self.guess_idxs])
            self.n_models = len(lhs_candidate_inds)
        else:
            lhs_candidate_inds = set(list(self.guess_idxs))
            self.n_models = len(lhs_candidate_inds)

        # Choose the guess column and run the regression
        for n in range(self.n_models):
            if shuffle:
                lhs_guess_term_index = np.random.choice(list(lhs_candidate_inds))    # Choose a random column index
            else:
                lhs_guess_term_index = list(lhs_candidate_inds)[0]

            # Remove the guess function from the set, so it isn't chosen again in next iterations
            lhs_candidate_inds.remove(lhs_guess_term_index)

            print(f"########\t{n+1}\t########")
            self.create_models_for_lhs_guess(lhs_guess_term_index)
            print(f"########\t{n+1}\t########\n")

        t_end = time.time()
        print(f"Total runtime: {round(t_end-t_start, 2)} sec")
