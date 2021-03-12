import numpy as np
import pandas as pd
from utils.function_libraries import *
from utils.signal_processing import *
from utils.identification import PI_Identifier
from utils.visualization import *
from itertools import combinations
from scipy.spatial import distance_matrix
from collections import namedtuple
import matplotlib as mpl
from collections import Counter
import os
from sklearn.cluster import AgglomerativeClustering
import pickle

mpl.use('Qt5Agg')

dirname = '.' + os.sep + 'singlePendulumCart' + os.sep + 'results' + os.sep
filename = dirname + 'simdata.csv'

sim_data = pd.read_csv(filename)

Xclean = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1])
X = StateSignal(sim_data['t'], sim_data.iloc[:, 1:-1], relative_noise_power=(0.2, 0.05, 0.2, 0.05))
SigProc = SignalProcessor(X, kernel='flattop', kernel_size=255)

ax = X.x_clean.plot()
pd.DataFrame(SigProc.x).plot(ax=ax)
SigProc.x_filtered.plot(ax=ax)
(SigProc.x_filtered-Xclean.x).iloc[200:-200].plot(subplots=True)

X_filt = StateSignal(SigProc.t, SigProc.x_filtered)


dXnoise = StateDerivativeSignal(X)
dXclean = StateDerivativeSignal(Xclean)
dX = StateDerivativeSignal(X_filt)

ax = dXnoise.dx.plot()
dXclean.dx.plot(ax=ax)
dX.dx.plot(ax=ax)
(dXclean.dx - dX.dx).iloc[200:-200].plot(subplots=True)

u = ForcingSignal(sim_data['t'], sim_data.iloc[:, -1])

# state_data = X.x
state_data = X_filt.x
state_derivative_data = dX.dx
input_data = u.u


# state_data = Xclean.x
# state_derivative_data = dXclean.dx
# input_data = u.u

dim = state_data.shape[1]

state_derivative_data = state_derivative_data.iloc[:, dim//2:]  # Remove ambiguity between x3,x4 and dx1,dx2
#%%
# Build library with sums of angles (state vars 2 and 3) and its sines/cosines
trig_data = trigonometric_library(state_data.iloc[:, 1:dim//2])

# linear/angular accelerations -> second half of state var derivatives

#%%
trig_state_derivative = product_library(trig_data, state_derivative_data)
acceleration_data = trig_state_derivative

# Function library Theta
theta = pd.concat([state_data, state_derivative_data,
                   trig_data, trig_state_derivative], axis=1)

cutoff = 200
theta = theta.iloc[cutoff:-cutoff, :]
theta.plot(subplots=True, layout=(3,4))

corr = theta.corr()
plot_corr(corr, theta.columns)
plt.show()

# x4 = state_data.iloc[cutoff:-cutoff,3]
# dx2 = state_derivative_data.iloc[cutoff:-cutoff,1]
# plt.plot(x4, color='blue')
# plt.plot(dx2, color='red')
# plt.legend(['x4','dx2'])
# plt.show()

cachename = dirname + 'singlePendSolutionsNoisy0_1'
rewrite = True
if os.path.exists(cachename) and not rewrite:
    print("Retrieving solution from cache.")
    with open(cachename, 'rb') as f:
        models = pickle.load(f)
else:
    print("No solution in cache, calculating solution from scratch.")
    EqnIdentifier = PI_Identifier(theta)
    EqnIdentifier.set_thresh_range(lims=(0.0001, 2), n=20)
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
plot_implicit_sols(gsols, glhs_guess_str,
                   theta.columns[theta_terms_idx],
                   normalize=False, show_labels=True)

clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                     compute_full_tree=True, distance_threshold=1).fit(dist)
# Cluster the models according to the activation distance
# plot_dendrogram(clustering)
# plt.show()

labels = clustering.labels_
models['label'] = labels

label_counter = Counter(labels)
drop_idx = np.array([label_counter[lbl]<2 for lbl in models.label])
drop_idx = np.argwhere(drop_idx)[:, 0]
models.drop(drop_idx, axis=0, inplace=True)
models.sort_values('label', axis=0, inplace=True)

tmp = np.vstack(models.active.values)
gsols = np.vstack(models.sol.values)
theta_terms_idx = np.apply_along_axis(lambda col: np.any(col), 0, gsols)
gsols = gsols[:, theta_terms_idx]
glhs_guess_str = models.lhs.values

plot_implicit_sols(gsols, glhs_guess_str, theta.columns[theta_terms_idx],
                   normalize=False, show_labels=True)

