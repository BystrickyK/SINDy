
from sklearn import linear_model
import time
import numpy as np

def seq_thresh_ls(A, b, threshold=0.5, n=10, alpha=0.1, verbose=False):
    # Pick candidate functions using ridge regression & threshold
    ridge_model = linear_model.Ridge(alpha=alpha)
    ridge_model.fit(A.values, b.values)
    x = ridge_model.coef_

    ndims = b.shape[1]
    for ii in range(0, n):
        tic = time.time()
        idx_small = np.abs(x) < threshold  # find small coefficients
        x[idx_small] = 0  # and set them to 0

        for dim in range(0, ndims):
            idx_big = ~idx_small[dim, :]
            model = linear_model.Ridge(alpha=alpha)
            model.fit(A.values[:, idx_big], b.values[:, dim])
            x[dim, idx_big] = model.coef_

        if verbose:
            toc = time.time()
            print('Iteration {} finished\t#\tIteration runtime: {:0.2f}us\t#'.format(ii, (toc-tic)*10**3))

    return x