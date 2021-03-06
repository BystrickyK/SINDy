import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import *
from scipy.spatial import distance_matrix
import matplotlib
import pickle
import os
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

matplotlib.use('Qt5Agg')

dirname = '.' + os.sep + 'doublePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata.csv'
sim_data = pd.read_csv(filename)

X = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1], noise_power=0)
dX = StateDerivativeSignal(X)
u = ForcingSignal(sim_data['t'], sim_data.iloc[:, -1])

state_data = X.x
state_derivative_data = dX.dx
input_data = u.u

dim = state_data.shape[1]

# %%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
angle_sums = sum_library(state_data.iloc[:, 1:dim // 2], (-2, -1, 0, 1, 2))
trig_data = trigonometric_library(angle_sums)
trig_data, rmvd = remove_twins(trig_data)

# linear/angular velocities -> second half of state vars
velocity_data = state_data.iloc[:, -dim // 2:]
vel_sq_data = square_library(velocity_data)

# linear/angular accelerations -> second half of state var derivatives
acceleration_data = state_derivative_data.iloc[:, -dim // 2:]

# %%
trig_vel = product_library(trig_data, velocity_data)
trig_vel_sq = product_library(trig_data, vel_sq_data)
trig_accel = product_library(trig_data, acceleration_data)

# Function library Theta
theta = pd.concat([velocity_data, acceleration_data,
                   trig_data, trig_vel, trig_vel_sq, trig_accel], axis=1)


cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]
theta = theta.astype('float32')


corrmat = theta.corr()
plot_corr(corrmat, theta.columns, labels=False)

cachename = dirname + 'doublePendSolutions2'
rewrite = False
if os.path.exists(cachename) and not rewrite:
    print("Retrieving solution from cache.")
    with open(cachename, 'rb') as f:
        models = pickle.load(f)
else:
    print("No solution in cache, calculating solution from scratch.")
    EqnIdentifier = PI_Identifier(theta)
    EqnIdentifier.set_thresh_range(lims=(0.0001, 1), n=20)
    EqnIdentifier.create_models(n_models=theta.shape[1], iters=8, shuffle=False)
    models = EqnIdentifier.all_models
    with open(cachename, 'wb') as f:
        pickle.dump(models, f)

# Remove duplicate models
sols = []
active = []
lhs_guess_str = []
residuals = []
model_hashes = []
for model in models:
    # 0 -> lhs string; 1 -> rhs string; 2 -> rhs solution; 3 -> complexity
    model_hash = hash(str(model[0]) + str(model[2]))
    if model_hash not in model_hashes:
        lhs_guess_idx = list(theta.columns).index(model[0])
        full_sol = list(model[2])
        full_sol.insert(lhs_guess_idx, -1)
        active_regressors = np.array(full_sol) != 0
        sols.append(full_sol)
        active.append(active_regressors)
        lhs_guess_str.append(model[0])
        residuals.append(model[4])
        model_hashes.append(model_hash)
    else:
        pass
models = pd.DataFrame([*zip(lhs_guess_str, sols, active, residuals)])
models.columns = ['lhs', 'sol', 'active', 'ssr']

# sol_print = lambda idx: print('{}\n = 0'.format(" + \n".join([str(param) + term for param, term in np.array([*zip(sols[idx].round(5), theta.columns)])[active[idx]]])))

sols = np.array(sols)
dist = distance_matrix(sols!=0, sols!=0, p=1)
models['dist'] = dist.tolist()
plot_activation_dist_mat(dist, lhs_guess_str)
plot_implicit_sols(sols, lhs_guess_str, theta.columns, normalize=False)

# Find consistent implicit models by finding 0s on the lower triangular distance matrix
dist_lt = np.tril(dist+100, -1)-100
idx_full = np.array(np.where(dist_lt==0))
idx = list(set(np.reshape(idx_full, -1)))

# Filter models for consistent models and plot
gsols = sols[idx]
theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
gsols = gsols[:, theta_terms_idx]
glhs_guess_str = np.array(lhs_guess_str)[idx]
gdist = distance_matrix(gsols!=0, gsols!=0, p=1)
plot_activation_dist_mat(gdist, glhs_guess_str)
plot_implicit_sols(gsols, glhs_guess_str, theta.columns[theta_terms_idx], normalize=False)

clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                   compute_full_tree=True, distance_threshold=1).fit(dist)
# Cluster the models according to the activation distance
# plot_dendrogram(clustering)
# plt.show()

labels = clustering.labels_
models['label'] = labels

label_counter = Counter(labels)
drop_idx = np.array([label_counter[lbl]<5 for lbl in models.label])
drop_idx = np.argwhere(drop_idx)[:, 0]
models.drop(drop_idx, axis=0, inplace=True)
models.sort_values('label', axis=0, inplace=True)

tmp = np.vstack(models.active.values)
gdist = distance_matrix(tmp, tmp, p=1)
gsols = np.vstack(models.sol.values)
theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
gsols = gsols[:, theta_terms_idx]
glhs_guess_str = models.lhs.values

plot_activation_dist_mat(gdist, glhs_guess_str)
plot_implicit_sols(gsols, glhs_guess_str, theta.columns[theta_terms_idx], normalize=False)

