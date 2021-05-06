from utils.function_libraries import *
import numpy as np
import pandas as pd

def create_basis_singPend(data):
    # identity = pd.Series((data['u'].values*0+1).T[0], data['u'].index, name='1')
    trig_basis_part = trigonometric_library(data['X'].iloc[:, 1])
    theta_basis = pd.concat([data['X'].iloc[:, [2,3]], trig_basis_part, data['u'],
                             data['DX'].iloc[:, [2,3]]], axis=1)
    return theta_basis

def create_basis_doubPend(data):
    trig_basis_part = trigonometric_library(data['X'].iloc[:, [1, 2]])
    theta_basis = pd.concat([data['X'].iloc[:, [3, 4, 5]], trig_basis_part, data['u'],
                             data['DX'].iloc[:, [3, 4, 5]]], axis=1)
    return theta_basis


def drop_bad_terms(theta):
    bad_idx = np.array([False for i in theta.columns])
    for i, term in enumerate(theta.columns):
        multiplicands = term.split('*')
        unique_terms = list(set(multiplicands))
        unique_term_occurences = np.array([np.sum([term in mult for mult in multiplicands]) for term in unique_terms])
        terms_occurences = dict(zip(unique_terms, unique_term_occurences))
        if np.any(unique_term_occurences>2): # If any sub-term occurs more than two times
            bad_idx[i] = True
            continue
        if len(unique_terms)>3: # if there's more than 3 unique sub-terms in the term
            bad_idx[i] = True
            continue
        if (('x_3' in unique_terms and 'u' in unique_terms) or
                ('x_4' in unique_terms and 'u' in unique_terms) or
                ('dx_3' in unique_terms and 'u' in unique_terms) or
                ('dx_4' in unique_terms and 'u' in unique_terms) or
                ('dx_3' in unique_terms and 'dx_4' in unique_terms) or
                ('x_3' in unique_terms and 'dx_4' in unique_terms) or
                ('x_3' in unique_terms and 'dx_3' in unique_terms) or
                ('x_4' in unique_terms and 'dx_4' in unique_terms) or
                ('x_4' in unique_terms and 'dx_3' in unique_terms) or
                ('x_4' in unique_terms and 'x_3' in unique_terms)):
            bad_idx[i] = True
            continue
            # if sin(x_2) occurs more than once OR
            # if there are more than 2 trig sub-terms at once OR
            # if there are two or more occurences of u
        if ((terms_occurences.get('sin(x_2)', False))>1 or
                (terms_occurences.get('sin(x_2)', 0) + terms_occurences.get('cos(x_2)', 0))>2 or
                (terms_occurences.get('u', 0))>1 or
                (terms_occurences.get('dx_3', False))>1 or
                (terms_occurences.get('dx_4', False))>1):
            bad_idx[i] = True
            continue

    print(f'{np.sum(bad_idx)}/{len(theta.columns)})')
    theta = theta.iloc[:, ~bad_idx]
    return theta
