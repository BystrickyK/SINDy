from sklearn import linear_model
import time
import numpy as np
import warnings



def seq_thresh_ls(A, b, threshold=0.5, n=10, alpha=0.1, verbose=False):
    # Pick candidate functions using ridge regression & threshold
    # ridge_model = linear_model.Ridge(alpha=alpha)
    # ridge_model.fit(A.values, b.values)
    # x = ridge_model.coef_
    x = np.linalg.lstsq(A, b)[0]

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
            x[idx_big] = np.linalg.lstsq(A.values[:, idx_big], b.values)[0]
        except:
            valid = False

        if verbose:
            toc = time.time()
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}us\t#'.format(ii, (toc - tic) * 10 ** 3))
            active_terms = np.sum(~idx_small)
            print("Number of active terms: {}/{}\n".format(active_terms, np.product(idx_small.shape)))

    x = np.array(x)
    return x, valid
