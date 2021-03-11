from itertools import combinations_with_replacement, chain, product, combinations
import pandas as pd
import numpy as np

# Returns a dictionary with all symbolic sum combinations
def sym_sum_library(column_names, multipliers=(-2, -1, 0, 1, 2)):

    multipliers = tuple(multipliers)
    subexpressions = [[(mult,expr) for mult in multipliers] for expr in column_names]

    sums = list([*product(*subexpressions)])

    bad_sums_idx = []
    for j, sum in enumerate(sums):
        mults = np.array([int(term[0]) for term in sum])
        exprs = np.array([term[1] for term in sum])

        # if all multipliers are 0
        if (np.sum(np.abs(mults)) == 0 or
                ((np.sum(np.abs(mults)) == np.abs(np.sum(mults))) and np.count_nonzero(mults)>1)):     # => if all parameter have the same signs
            # np.count_nonzero(params) == 1 and (np.sum(params) < 0 or np.sum(params) == 2) or
            # np.all(params < 0) or
            # np.count_nonzero(params<0) == 2):
            bad_sums_idx.append(j)
        else:
            nonzero_idx = mults!=0
            sums[j] = [*zip([*mults[nonzero_idx]], [*exprs[nonzero_idx]])]

    sums = np.delete(sums, bad_sums_idx, axis=0)

    return sums

# Returns a dictionary with all symbolic polynomials of requested orders
def sym_poly_library(column_names, poly_orders=(1, 2)):
    polynomials = {}  # Dictionary of all requested symbolic polynomials

    # If only one polynomial order is requested
    if isinstance(poly_orders, (int, float)):
        poly_orders = int(poly_orders)
        polynomials['poly' + str(poly_orders)] = \
            [*combinations_with_replacement(column_names, poly_orders)]

    elif isinstance(poly_orders, (list, np.ndarray, tuple)):
        poly_orders = tuple(poly_orders)
        for polynomial_order in poly_orders:
            poly = combinations_with_replacement(column_names, polynomial_order)
            polynomials['poly' + str(polynomial_order)] = [*poly]

    else:
        raise TypeError("\'poly_orders\' is expected to be of type int|list|np.array|tuple")

    return polynomials


# Takes all values from a dictionary and returns a flat list of them
def flatten_dict(dictionary):
    dict_vals = dictionary.values()
    dict_flat_iterator = chain.from_iterable(dict_vals)
    return tuple([*dict_flat_iterator])


# Returns vectors of values of requested polynomials as a pandas DataFrame
def sum_library(x, multipliers=(-2, -1, 0, 1, 2)):
    sym_sums = sym_sum_library(x.columns, multipliers)

    samples = x.shape[0]
    # TODO: The inner for loop can probably be replaced with a vectorized solution
    # TODO: to run faster
    fun_lib = np.zeros(shape=(samples, len(sym_sums)))
    for col, sym_sum in enumerate(sym_sums):
        for mult, sym_var in sym_sum:
            fun_lib[:, col] += float(mult) * x[sym_var]

    sum_str = [' + '.join([str(mult) + '*' + str(var) for mult,var in subexpr]) for subexpr in sym_sums]
    df = pd.DataFrame(
        data=fun_lib,
        columns=sum_str
    )
    return df

def trigonometric_library(x):
    sines = np.sin(x)
    sines.columns = ['sin(' + col_name + ')' for col_name in x.columns]

    cosines = np.cos(x)
    cosines.columns = ['cos(' + col_name + ')' for col_name in x.columns]

    df = pd.concat([sines, cosines], axis=1)
    return df

def product_library(multipliers, multiplicands):
    samples = multipliers.shape[0]
    terms = [*product(multipliers, multiplicands)]
    fun_lib = np.zeros(shape=(samples, len(terms)))
    for col, term in enumerate(terms):
        fun_lib[:, col] = multipliers[term[0]] * multiplicands[term[1]]

    column_str = [str(i) + '*' + str(j) for i,j in terms]
    df = pd.DataFrame(fun_lib)
    df.columns = column_str
    return df

def square_library(terms):
    samples = terms.shape[0]
    dims = terms.shape[1]
    fun_lib = np.zeros(shape=(samples, dims))
    term_str = terms.columns
    for col, term in enumerate(term_str):
       fun_lib[:, col] = np.square(terms[term])

    df = pd.DataFrame(fun_lib)
    df.columns = [str(i) + '*' + str(i) for i in term_str]
    return df

# Returns vectors of values of requested polynomials as a pandas DataFrame
def poly_library(x, poly_orders=(1, 2)):
    sym_polys = sym_poly_library(x.columns, poly_orders)
    sym_polys = flatten_dict(sym_polys)

    samples = x.shape[0]
    # TODO: The inner for loop can probably be replaced with a vectorized solution
    # TODO: to run faster
    fun_lib = np.ones(shape=(samples, len(sym_polys) + 1))
    for col, sym_pol in enumerate(sym_polys, start=1):
        for sym_var in sym_pol:
            fun_lib[:, col] *= x[sym_var]

    df = pd.DataFrame(
        data=fun_lib,
        index=x.index,
        columns=['1'] + ['*'.join(poly) for poly in sym_polys]
    )
    return df


def remove_twins(fun_lib):
    corr = fun_lib.corr()
    corr_lower_tri = np.tril(corr, k=-1)
    lib_identity_idx = np.abs(corr_lower_tri) == 1
    lib_identity_idx = np.array(np.nonzero(lib_identity_idx))[1,:]
    lib_identity_labels = fun_lib.columns[lib_identity_idx]
    lib_data_without_identities = fun_lib.drop(lib_identity_labels, axis=1)
    return lib_data_without_identities, lib_identity_labels
