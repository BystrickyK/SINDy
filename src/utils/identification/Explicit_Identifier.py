import numpy as np
from utils.control_structures import timeout
from utils.regression import *
from utils.model_selection import *
from collections import namedtuple
import pandas as pd
# from dask import delayed
import time

class Explicit_Identifier:
    def __init__(self, theta_train, target, iters=10, verbose=False):
        """
        Takes in the regressor matrix and identifies potential state potential
        models by guessing active regressors and varying regression hyperparameters
        Args:
            theta: Function library matrix
        """
        self.theta_train = theta_train
        self.theta_val = theta_train
        self.target = target
        self.results = []

        self.verbose = verbose
        self.iters = iters

        eqn = {'guess_function_string': [],
               'xi': [],
               'condition_num': [],
               'train_metric': [],
               'val_metric': []}
        # Initialize empty dataframe
        self.models = pd.DataFrame(eqn)

        self.weights = None
        self.thresh = None

        print("Created new identifier object.\n")

    def set_thresh_range(self, lims=(0.01, 2), n=30):
        thresh = np.linspace(lims[0], lims[1], n, endpoint=True)
        self.thresh = thresh

    # def create_models(self, iters=8):
    #     """
    #     iters: How many iterations should the SQLS algorithm do.
    #     """
    #     t_start = time.time()
    #
    #     self.iters = iters
    #
    #     print('a')
    #     for i in range(self.xdot.shape[1]):
    #         target = self.xdot.iloc[:, i]
    #         print(f"########\t{i+1}\t########")
    #         self.create_models_for_target(target)
    #         print(f"########\t{i+1}\t########\n")
    #
    #     t_end = time.time()
    #     print(f"Total runtime: {round(t_end-t_start, 2)} sec")

    def create_models(self):
        """
        Given the target data, returns a set of models
        """
        print(f"########\tIdentifying: {self.target.name}\t########")

        lowest_error = 1e9
        for i, hyperparameter in enumerate(self.thresh, start=1):
            mdl = self.create_model(self.target, hyperparameter)
            if mdl['val_metric'] < lowest_error:  # track lowest error model
                lowest_error = mdl['train_metric']

        run_info = ("Created models for {tar}\n\tLowest error: {error}".format(
            tar=self.target.name,
            error=round(lowest_error, 3)))

        print(run_info)

    def create_model(self, target, hyperparameter):
        """
        Given a target column and hyperparameters, runs the regression and returns the resulting model.
        """
        i_start = time.time()

        # Find sparse solution using STLQ
        x, residuals, valid, condition_num = sequentially_thresholded_least_squares(A=self.theta_train, b=target, n=self.iters,
                                                                  lambda_=hyperparameter, verbose=True)
        tmp = self.theta_train@x - target
        plt.figure()
        plt.plot(self.theta_train.loc[:,'x_2'], linewidth=1.5, color='tab:blue')
        plt.plot(target, '--', linewidth=1.5, color='tab:red')
        print('.')
        # x, valid, residuals = seq_energy_thresh_ls(A_train=theta_train, b_train=target_function,
        #                                                  n=iters, lambda_=hyperparameter, verbose=False)

        # x, valid, condition_num = sequentially_energy_thresholded_least_squares(A=theta_train, b=target_function,
        #                                                                  weights=None, target_str=self.target, n=self.iters,
        #                                                                  lambda_=hyperparameter, verbose=False)
        # Normalize the solution so its L2 norm is 100

        train_metric = np.sqrt(np.sum(np.square(residuals)))
        mdl = {'guess_function_string': target.name,
               'xi': x,
               'condition_num': condition_num,
               'train_metric': train_metric,
               'val_metric': train_metric}
        self.models = self.models.append(mdl, ignore_index=True)

        i_end = time.time()

        if self.verbose:
            nnz_idx = np.nonzero(x)[0]
            x_str = ["{:3f}".format(x[i]) + '*' + self.theta_train.columns[i] for i in nnz_idx]
            x_str = " + ".join(x_str)
            print('Runtime:\t\t{:0.2f}ms\nComplexity:\t\t{}\nRHS:\t\t{}\nFit:\t\t{}'.format(
                (i_end - i_start) * 10 ** 3, mdl['complexity'], x_str, train_metric))

        return mdl
