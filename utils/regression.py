from sklearn.linear_model import Ridge
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from copy import copy

def seq_thresh_ls(A, b, threshold=0.5, n=10, alpha=0.1, verbose=False):
    # Pick candidate functions using ridge regression & threshold
    # ridge_model = linear_model.Ridge(alpha=alpha)
    # ridge_model.fit(A.values, b.values)
    # x = ridge_model.coef_
    x = np.linalg.lstsq(A, b)[0]
    residuals = None
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
        try:
            # model = linear_model.Ridge(alpha=alpha)
            # model.fit(A.values[:, idx_big], b.values)
            # x[idx_big] = model.coef_
            x[idx_big], residuals = np.linalg.lstsq(A.values[:, idx_big], b.values)[0:2]
            residuals = residuals[0]
        except:
            valid = False

        if verbose:
            toc = time.time()
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}us\t#'.format(ii, (toc - tic) * 10 ** 3))
            active_terms = np.sum(~idx_small)
            print("Number of active terms: {}/{}\n".format(active_terms, np.product(idx_small.shape)))

    x = np.array(x)
    return x, valid, residuals

def seq_energy_thresh_ls(A, b, lambda_=0.05, n=10, verbose=False):
    # Pick candidate functions using sequentially energy-thresholded least squares
    x, _, _, singvals = np.linalg.lstsq(A, b, rcond=None)

    residuals = None
    valid = True
    idx_big = np.ones([A.shape[1],], int)
    # ndims = b.shape[1]
    for ii in range(0, n):
        tic = time.time()
        # Hadamard product (element-wise multiplication); x.T extends row-wise into A's shape (broadcasting)
        terms = (A * x.T).values
        term_energies = np.sum(np.square(terms), axis=0)
        energy_thresh = term_energies.max() * lambda_
        Aw = A.values[:, idx_big]  # Only pick terms that haven't been thresholded
        bw = b.values
        # Calculate squared error signal
        error = np.sum(terms[:, idx_big], axis=1) - bw
        # Set higher weights on samples with high error
        weights = error**2 + 1
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

        model = Ridge(alpha=0.5)
        try:
            # x[idx_big], residuals, rank, singvals = np.linalg.lstsq(Aw, bw, rcond=None)
            model.fit(Aw, bw, sample_weight=weights)
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
