import pandas as pd
import numpy as np
from utils.visualization import *
from scipy.spatial import distance_matrix as dist_mat
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

def process_models(models, theta_cols):
    #%% Remove duplicate models and reorder the equations to implicit
    sols = []
    active = []
    lhs_guess_str = []
    residuals = []
    trainfits = []
    valfits = []
    model_hashes = []
    for model in models:
        # 0 -> lhs string; 1 -> rhs string; 2 -> rhs solution; 3 -> complexity
        # 4 -> residuals; 5 -> trainfit
        model_hash = hash(str(model[0]) + str(model[2]))
        if model_hash not in model_hashes:
            lhs_guess_idx = list(theta_cols).index(model[0])
            full_sol = list(model[2])
            full_sol.insert(lhs_guess_idx, -1)
            full_sol = np.array(full_sol)
            full_sol = full_sol / np.linalg.norm(full_sol) * 100 # Normalize the parameters so their L2 norm is 100
            active_regressors = np.array(full_sol) != 0
            sols.append(full_sol)
            active.append(active_regressors)
            lhs_guess_str.append(model[0])
            residuals.append(model[4])
            trainfits.append(model[5])
            valfits.append(model[6])
            model_hashes.append(model_hash)
        else:
            pass
    models = pd.DataFrame([*zip(lhs_guess_str, sols, active, residuals, trainfits, valfits)])
    models.columns = ['lhs', 'sol', 'active', 'ssr', 'trainerror', 'valerror']
    return models

def distance_matrix(models, plot=False):
    #%% Visualize the solutions -> calculate and plot activation distance matrix
    # and plot the matrix of implicit solutions
    sols = np.vstack(models['sol'].values)
    dist = dist_mat(sols!=0, sols!=0, p=1)
    if plot:
        lhs = models['lhs'].values
        plot_activation_dist_mat(dist, lhs)
    return dist

def consistent_models(models, dist=None, distance_threshold=0.1, min_cluster_size=3):
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

    return models
