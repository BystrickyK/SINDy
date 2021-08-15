import pandas as pd
import numpy as np
from utils.visualization import *
from utils.model_selection import *
from utils.control_structures import time_symbol
from scipy.spatial import distance_matrix as dist_mat
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import sympy as sp
from sympy.utilities.codegen import codegen

def model_unique(models):

    #%% Remove duplicate models
    model_hashes=[]
    drop_row_idx = []
    for idx, model in models.iterrows():
        # Create model hash using the model's guess candidate function and the model's parameter vector
        model_hash = hash(str(model['guess_function_string']) + str(model['xi']))
        if model_hash not in model_hashes:
            model_hashes.append(model_hash)
            drop_row_idx.append(False)
        else:
            drop_row_idx.append(True)

    drop_row_idx = np.array(drop_row_idx)
    unique_models = models.iloc[~drop_row_idx, :]
    unique_models = unique_models.reset_index(drop=True)
    return unique_models

def distance_matrix(models, plot=False):
    #%% Visualize the solutions -> calculate and plot activation distance matrix
    # and plot the matrix of implicit solutions
    sols = np.vstack(models['xi'].values)
    dist = dist_mat(sols!=0, sols!=0, p=1)
    if plot:
        lhs = models['lhs'].values
        plot_activation_dist_mat(dist, lhs)
    return dist

def model_consistent(models, dist=None, distance_threshold=0.1, min_cluster_size=3):
    if dist is None:
        dist = distance_matrix(models, plot=False)

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         compute_full_tree=True, distance_threshold=distance_threshold).fit(dist)

    labels = clustering.labels_
    # Add a cluster label to each model
    models['label'] = labels

    # Calculate the sizes of clusters
    label_counter = Counter(labels)

    # Find indices of models that are in a cluster with less than 3 points
    drop_idx = np.array([label_counter[lbl]<min_cluster_size for lbl in models.label])
    drop_idx = np.argwhere(drop_idx)[:, 0]

    # Drop inconsistent models
    models.drop(drop_idx, axis=0, inplace=True)
    models.sort_values('label', axis=0, inplace=True)
    models = models.reset_index(drop=True)

    return models

def model_sparse(models, threshold=10):
    models = models[ models['n_terms'] < threshold ] # extract sparse models
    return models

def model_activations(models):
    models['active'] = models.apply(lambda row: np.abs(row['xi'])>0, axis=1)
    models['n_terms'] = models.apply(lambda row: np.sum(np.array(row['active'])), axis=1)
    return models

def model_val_rmse(models, theta_val):
    val_metrics = []
    for i, model in models.iterrows():
        xi = model['xi']
        rmse = calculate_rmse(theta_val, xi, 0)
        val_metrics.append(rmse)
    models['validation_metric'] = val_metrics
    return models

def model_aic(models, theta):
    models['aic'] = models.apply(lambda row: calculate_aic(theta, row['xi'], 0),
                             axis=1)
    models['aic'] = models['aic'] - models['aic'].min()
    return models

# def model_sparse(models, threshold):
#     sparse_idx =

def model_equation_strings(models, theta_cols):

    def equation_string(model):

        # Extract active terms
        active = model['active']

        params = model['xi'][active]
        signs = np.sign(params)
        params = np.abs(params)
        signs = ['+' if sign==1 else '-' for sign in signs]

        functions = theta_cols[active]

        # Create param-function pair tuples
        term_pairs = zip(signs,
                         np.round(np.abs(params), 5),
                         functions)
        term_pairs = [*term_pairs]

        # Create the equation string
        eqn_str = [' ' + sgn + ' ' + str(par) + '*' + term for sgn, par, term in term_pairs]
        eqn_str = ''.join(eqn_str)
        return eqn_str

    models['eqn_str'] = models.apply(lambda row: equation_string(row), axis=1)

    return models

def model_symbolic_implicit_eqns(models, lhsvar):
    lhssymvar = sp.parse_expr(lhsvar)

    models['eqn_sym_implicit'] = models.apply(lambda row: sp.parse_expr(row['eqn_str']),
                                              axis=1)

    # Drop models that don't contain the lhs variable (target variable)
    keep_idx = []
    for i, mdl in models.iterrows():
        if mdl['eqn_sym_implicit'].has(lhssymvar):
            keep_idx.append(i)

    keep_idx = np.array(keep_idx)
    models = models.loc[keep_idx, :]
    models = models.reset_index(drop=True)

    return models

def model_symbolic_eqn(models, lhsvar):
    lhs_symvar = sp.parse_expr(lhsvar)

    def solve_eqn(mdl):
        sol = sp.solve(mdl['eqn_sym_implicit'], lhs_symvar)[0]
        print(time_symbol())
        print(sol)
        return sol

    models['eqn_sym'] = models.apply(lambda row: solve_eqn(row),
                                     axis=1)
    return models

def model_lambdify_eqn(models, vars):
    symvars = [sp.parse_expr(var) for var in vars]

    def lambdify(symeqn):
        eqn_lambda_ = sp.lambdify(symvars, symeqn, {'sgn':np.sign})
        eqn_lambda = lambda input: eqn_lambda_(*input)
        return eqn_lambda

    models['eqn_lambda'] = models.apply(lambda row: lambdify(row['eqn_sym']), axis=1)
    return models


def model_validate(models, theta):
    def calc_rmse(xi):
        error = np.dot(theta, xi)
        rmse = np.sqrt(np.mean(np.square(error)))
        return rmse

    models['rmse'] = models.apply(lambda row: calc_rmse(row['xi']), axis=1)
    return models

def model_validate(models, val_data_xu, val_data_dx):
    def calc_rmse(model):
        preds = np.apply_along_axis(model, axis=1, arr=val_data_xu)
        error = preds - val_data_dx
        rmse = np.sqrt(np.mean(np.square(error)))
        return rmse

    models['rmse'] = models.apply(lambda row: calc_rmse(row['eqn_lambda']), axis=1)
    return models
