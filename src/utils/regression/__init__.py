from sklearn.linear_model import Ridge
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
import warnings
from copy import copy

warnings.filterwarnings("ignore")

def sequentially_thresholded_least_squares(A, b, weights=None, threshold=0.5, n=10, verbose=False):
    # Pick candidate functions using ridge regression & threshold
    ridge_model = Ridge(alpha=0.01)
    ridge_model.fit(A.values, b.values)
    x = ridge_model.coef_

    if isinstance(weights, (list, np.ndarray)):
        if len(weights) != len(b):
            raise ValueError("The length of the weight vector isn't equal to the length of the target vector")

    valid = True
    # ndims = b.shape[1]
    for ii in range(0, n):
        tic = time.time()
        idx_small = np.abs(x) < threshold  # find small coefficients
        x[idx_small] = 0  # and set them to 0

        # for dim in range(0, ndims):
        idx_big = ~idx_small
        if sum(idx_big) == 0:
            valid = False
            # print("All candidate functions in dimension {} got thresholded.\nConsider decreasing the "
            #                  "thresholding value.")

        model = Ridge(alpha=0.01)
        try:
            # x[idx_big], residuals, rank, singvals = np.linalg.lstsq(Aw, bw, rcond=None)
            model.fit(A, b, sample_weight=weights)
            x[idx_big] = model.coef_
            residuals = residuals[0]
        except:
            valid = False

        if verbose:
            toc = time.time()
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}ms\t#'.format(ii, (toc - tic) * 10 ** 3))
            active_terms = np.sum(~idx_small)
            print("Number of active terms: {}/{}".format(active_terms, np.product(idx_small.shape)))


    x = np.array(x)
    return x, valid, residuals

def sequentially_energy_thresholded_least_squares(A, b, weights=None, target_str=None,
                                                  lambda_=0.05, n=10, verbose=False):

    # Instead of thresholding out candidate functions based on their coefficient value,
    # threshold based on their *energy* => if a function has low energy, set coefficient to 0

    # Pick candidate functions using sequentially energy-thresholded least squares
    model = Ridge(alpha=1)
    model.fit(A, b)
    x = model.coef_

    target = np.array([col==target_str for col in A.columns])

    valid = True
    idx_small = np.zeros([A.shape[1],], int)
    idx_big = ~idx_small
    # ndims = b.shape[1]
    for ii in range(0, n):
        tic = time.time()

        Aw = A.values[:, idx_big]  # Only pick terms that haven't been thresholded
        bw = b.values

        # Hadamard product (element-wise multiplication); x.T extends row-wise into A's shape (broadcasting)
        terms = (A * x.T).values
        term_energies = np.sum(np.square(terms), axis=0)
        energy_thresh = term_energies.max() * lambda_

        idx_small = (term_energies < energy_thresh)  # find low energy terms
        if target is not None:
            # idx_small = np.all([idx_small, ~target], axis=)
            idx_small = np.logical_and(idx_small, ~target)
        x[idx_small] = 0  # and set their contributions to 0

        # for dim in range(0, ndims):
        idx_big = ~idx_small
        if sum(idx_big) == 0:
            valid = False
            break

        # fig, axs = plt.subplots(nrows=3, ncols=1, tight_layout=True)
        # axs[0].plot(weights)
        # axs[0].set_title(f"{np.sum(idx_big)}\nFuture weights")
        # axs[1].plot(error)
        # axs[1].set_title("Current Error")
        # axs[2].bar(range(len(x)), x)
        # axs[2].set_title("Current Parameters")
        # plt.show()

        model = Ridge(alpha=0.01, fit_intercept=False, tol=1e-4)
        try:
            # x[idx_big], residuals, rank, singvals = np.linalg.lstsq(Aw, bw, rcond=None)
            # print(f'{np.max(weights)}, {np.min(weights)}')
            model.fit(Aw, bw, sample_weight=weights)
            x[idx_big] = model.coef_
        # except LinAlgWarning:
        #     print("Poorly conditioned")
        except:
            valid = False

        if verbose:
            toc = time.time()
            print(A.columns[target])
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}ms\t#'.format(ii, (toc - tic) * 10 ** 3))
            active_terms = np.sum(~idx_small)
            print("Number of active terms: {}/{}".format(active_terms, np.product(idx_small.shape)))

    x = np.array(x)
    return x, valid, np.linalg.cond(Aw)


def seq_energy_thresh_ls_val(A_train, b_train, A_val, b_val,
                             lambda_=0.05, n=10, verbose=False):
    # Pick candidate functions using sequentially energy-thresholded least squares
    model = Ridge(alpha=0.1)
    model.fit(A_train, b_train)
    x = model.coef_

    residuals = None
    valid = True
    idx_small = np.zeros([A_train.shape[1],], int)
    idx_big = ~idx_small
    # ndims = b.shape[1]
    for ii in range(0, n):
        tic = time.time()

        Aw = A_train.values[:, idx_big]  # Only pick terms that haven't been thresholded
        bw = b_train.values

        # Hadamard product (element-wise multiplication); x.T extends row-wise into A's shape (broadcasting)
        terms = (A_val * x.T).values
        term_energies = np.sum(np.square(terms), axis=0)
        energy_thresh = term_energies.max() * lambda_

        idx_small = (term_energies < energy_thresh)  # find low energy terms
        x[idx_small] = 0  # and set their contributions to 0

        # for dim in range(0, ndims):
        idx_big = ~idx_small
        if sum(idx_big) == 0:
            valid = False
            break

        # fig, axs = plt.subplots(nrows=3, ncols=1, tight_layout=True)
        # axs[0].plot(weights)
        # axs[0].set_title(f"{np.sum(idx_big)}\nFuture weights")
        # axs[1].plot(error)
        # axs[1].set_title("Current Error")
        # axs[2].bar(range(len(x)), x)
        # axs[2].set_title("Current Parameters")
        # plt.show()

        model = Ridge(alpha=1, fit_intercept=False, tol=1e-4)
        try:
            # x[idx_big], residuals, rank, singvals = np.linalg.lstsq(Aw, bw, rcond=None)
            # print(f'{np.max(weights)}, {np.min(weights)}')
            model.fit(Aw, bw)
            x[idx_big] = model.coef_
        # except LinAlgWarning:
        #     print("Poorly conditioned")
        except:
            valid = False

        if verbose:
            toc = time.time()
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}ms\t#'.format(ii, (toc - tic) * 10 ** 3))
            active_terms = np.sum(~idx_small)
            print("Number of active terms: {}/{}".format(active_terms, np.product(idx_small.shape)))

    x = np.array(x)
    return x, valid, np.linalg.cond(Aw)
