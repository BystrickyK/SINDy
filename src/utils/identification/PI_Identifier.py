import numpy as np
from utils.control_structures import timeout
from utils.regression import *
from utils.model_selection import *
from collections import namedtuple
import pandas as pd
# from dask import delayed
import time

class PI_Identifier:
    def __init__(self, theta, verbose=False):
        """
        Takes in the regressor matrix and identifies potential state potential
        models by guessing active regressors and varying regression hyperparameters
        Args:
            theta: Function library matrix
        """
        self.theta = theta

        self.verbose = verbose

        # Initialize empty dataframe for models
        self.models = pd.DataFrame()

        self.target_function_string = None
        self.target_function_idx = None
        self.weights = None
        self.thresh = None

        self.guess_idxs = None
        
    def set_thresh_range(self, lims=(0.01, 0.8), n=10):
        thresh = np.linspace(lims[0], lims[1], n, endpoint=True)
        self.thresh = thresh

    def set_weights(self, weights):
        # not implemented
        self.weights = weights

    def set_target(self, target):
        self.target_function_idx = np.where(self.theta.columns == target)[0][0]
        self.target_function_string = target

    def set_guess_cols(self, guess_cols):
        if not isinstance(guess_cols, list):
            guess_cols = [guess_cols]

        if isinstance(guess_cols[0], str):
            self.guess_idxs = []
            for guess_str in guess_cols:
                guess_idx = np.argwhere(self.theta.columns==guess_str)[0][0]
                self.guess_idxs.append(guess_idx)
        else:
            self.guess_idxs = guess_cols

    def create_model(self, guess_index, hyperparameter):
        """
        Given a guess index and hyperparameters, runs the regression and returns the resulting model.
        """

        target_function = self.theta.iloc[:, guess_index]  # theta_i -> guess candidate function
        function_lib = self.theta.drop(target_function.name, axis=1)  # Theta_i -> candidate function library - theta_i

        xi, valid, condition_num = sequentially_energy_thresholded_least_squares(A=function_lib, b=target_function,
                                                                         weights=None, target_str=self.target_function_string, n=self.iters,
                                                                         lambda_=hyperparameter, verbose=False)

        # Impute the guess term parameter into the solution vector
        xi = np.insert(xi, guess_index, -1)

        # Normalize the solution so the target function coefficient is 1
        if self.target_function_string is not None:
            xi = xi / xi[self.target_function_idx]
        else:
            # Normalize the solution so its L2 norm is 100
            xi = xi / np.linalg.norm(xi)

        train_metric = calculate_rmse(self.theta.values, xi, 0)
        mdl = {'guess_function_string': target_function.name,
               'xi': xi,
               'complexity': np.sum(np.nonzero(xi)),
               'condition_num': condition_num,
               'train_metric': train_metric}

        self.models = self.models.append(mdl, ignore_index=True)

        return mdl

    def create_models_for_lhs_guess(self, guess_function_index):
        """
        Given the guess column index, creates a model for each set of hyperparameters. The number of iterations
        depends on the length of self.thresh
        """

        lowest_error = 1e9
        for i, hyperparameter in enumerate(self.thresh, start=1):
            mdl = self.create_model(guess_function_index, hyperparameter)
            # print(f"{mdl['val_metric']}, {mdl['train_metric']}")
            if mdl['train_metric'] < lowest_error:
                lowest_error = mdl['train_metric']

        run_info = ("Created models for #{idx} / {total}\n\tFor guess function: {guess}\n\tLowest error: {error}".format(
            idx=guess_function_index,
            total=self.n_models,
            guess=self.theta.columns[guess_function_index],
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
            self.n_models = self.theta.shape[1]

        # Set of possible LHS functions (defined as column indices from function library)
        if self.guess_idxs is None:
            lhs_candidate_inds = set([*range(len(self.theta.columns))])
        elif isinstance(self.guess_idxs, int):
            lhs_candidate_inds = set([self.guess_idxs])
            self.n_models = len(lhs_candidate_inds)
        else:
            lhs_candidate_inds = set(list(self.guess_idxs))
            self.n_models = len(lhs_candidate_inds)

        # Choose the guess column and run the regression
        for n in range(self.n_models):
            if shuffle:
                guess_function_index = np.random.choice(list(lhs_candidate_inds))    # Choose a random column index
            else:
                guess_function_index = list(lhs_candidate_inds)[0]

            # Remove the guess function from the set, so it isn't chosen again in next iterations
            lhs_candidate_inds.remove(guess_function_index)

            print(f"########\t{n+1}\t########")
            self.create_models_for_lhs_guess(guess_function_index)
            print(f"########\t{n+1}\t########\n")

        t_end = time.time()
        print(f"Total runtime: {round(t_end-t_start, 2)} sec")
