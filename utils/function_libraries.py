from itertools import combinations_with_replacement, chain, product
import pandas as pd
import numpy as np

# Returns a dictionary with all symbolic sum combinations
def sym_sum_library(column_names, multipliers=(-2, -1, 0, 1, 2)):

    multipliers = tuple(multipliers)
    subexpressions = [[(mult,expr) for mult in multipliers] for expr in column_names]

    sums = [*product(*subexpressions)]

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
            fun_lib[:, col] += mult * x[sym_var]

    sum_str = [' + '.join([str(mult) + '*' + str(var) for mult,var in subexpr]) for subexpr in sym_sums]
    df = pd.DataFrame(
        data=fun_lib,
        columns=sum_str
    )
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
