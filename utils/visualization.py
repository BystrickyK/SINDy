import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from seaborn import color_palette, diverging_palette
from datetime import datetime
from functools import wraps
import os
from scipy.cluster.hierarchy import dendrogram
from utils.tools import *
import time

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'
plt.style.use({'seaborn', './images/BystrickyK.mplstyle'})


def set_save_plot_options(save=False,
                          add_stamp=True, plot=True, format='.svg', dpi=300):

    # Move to project root
    # path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    path = os.getcwd()
    # Move to images
    path = path + os.sep + 'images'

    file_path = path + os.sep
    if add_stamp:
        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M")
        file_path = file_path + '_' + now

    if not os.path.exists(file_path):
        print(f"Path {file_path} doesn't exist. Attempting dir creation.")
        try:
            os.mkdir(file_path)
        except OSError:
            print(f"Creation of the directory {file_path} failed")
        else:
            print(f"Successfully created the directory {file_path}")
            path = file_path + os.sep
    else:
        print(f"Folder {file_path} already exists")
        path = file_path + os.sep


    def save_and_plot(save=save, plot=plot, filename='default', stamp=False):

        def decorator(fun):

            @wraps(fun)
            def wrapper(*args, **kwargs):
                fig = fun(*args, **kwargs)
                if save:
                    if stamp:
                        rand_stamp = str(time.time()).replace('.','_')
                        save_path = path + rand_stamp + '_' + filename + format
                    else:
                        save_path = path + filename + format
                    plt.savefig(save_path, dpi=dpi)

                if not plot:
                    plt.close(fig)

                return fig

            return wrapper

        return decorator

    return save_and_plot

####################################################################################
####################################################################################
save_and_plot = set_save_plot_options(save=True, plot=False,
                                      add_stamp=True, format='.jpg', dpi=180)
####################################################################################
####################################################################################

@save_and_plot(filename='vector', stamp=True)
def plot_tvector(t, X, var_name='x', title=None):
    dims = X.shape[1]

    colors = color_palette('dark')
    with plt.style.context({'./images/BystrickyK.mplstyle', 'seaborn'}):
        fig, axs = plt.subplots(dims, 1, figsize=(8, 8), tight_layout=True, sharex=True)
        if title == None:
            pass
        else:
            fig.suptitle(title)
        for ii, ax in enumerate(axs):
            ax.plot(t, X[:, ii], color=colors[ii])
            ylabel_str = rf'$ {var_name}_{ii+1} (t) $'
            ax.set_ylabel(ylabel_str)
        ax.set_xlabel(rf'$ Time \  t \  [s] $')
        plt.show()
    # return fig


def plot_ksi(ksi, theta, dx, ax, show_sparse=True, show_sparse_tol=0.1):
    ksi = ksi.T  # fix the ksi output shape
    if show_sparse:
        idx_dim_active = np.apply_along_axis(lambda x: sum(abs(x)) > show_sparse_tol, 1, ksi)
        ksi = ksi[idx_dim_active, :]
        theta = theta.iloc[:, idx_dim_active]

    colormap = color_palette('coolwarm', as_cmap=True)

    thetastr = parse_function_strings(theta.columns)
    dxstr = parse_function_strings(dx.columns)
    dxstr = parse_function_str_add_dots(dxstr)

    with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
        ax.matshow(ksi, cmap=colormap, vmin=-2, vmax=2)
        # plt.colorbar(p, ax=ax)
        ax.set_yticks([*range(min(theta.shape))])
        ax.set_yticklabels(thetastr)
        ax.yaxis.set_tick_params(rotation=30, labelsize=10)
        ax.set_xticks([*range(min(dx.shape))])
        ax.set_xticklabels(dxstr)
        ax.xaxis.set_tick_params(rotation=45, labelsize=10)
        ax.xaxis.set_tick_params(labelsize=15)
        for (col, row), val in np.ndenumerate(ksi):
            ax.text(row, col, '{:0.2f}'.format(val), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

@save_and_plot(filename='corr', stamp=True, plot=True)
def plot_corr(corr, regressor_names, labels=True):

    # labelstr = parse_function_strings(regressor_names)
    # labelstr = parse_function_str_add_dots(regressor_names)
    labelstr = regressor_names
    labelstr = np.array(d_to_dot(latexify(labelstr)))

    figsize = tuple(np.array(corr.shape) * 0.15 + 5)
    if figsize[0] > 70:
        figsize = tuple(np.array(figsize) * 0.25)
    print(f"Creating correlation matrix plot\nImage size:\t{figsize}")

    fig, ax = plt.subplots(1, 1,
                           figsize=figsize, tight_layout=True)
    with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
        im = ax.matshow(corr, cmap='viridis', vmin=-1, vmax=1)
        ax.set_yticks([*range(min(labelstr.shape))])
        ax.set_yticklabels(labelstr)
        ax.yaxis.set_tick_params(rotation=0, labelsize=10)
        ax.set_xticks([*range(min(labelstr.shape))])
        ax.set_xticklabels(labelstr)
        ax.xaxis.set_tick_params(rotation=90, labelsize=10)
        fig.colorbar(im, ax=ax)
        if labels:
            for (col, row), val in np.ndenumerate(corr):
                ax.text(row, col, '{:0.2f}'.format(val), ha='center', va='center',
                        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='0.3'))

@save_and_plot(filename='activation_dist', stamp=True, plot=True)
def plot_activation_dist_mat(dist_mat, lhs_guess_strings, labels=True):

    # labelstr = parse_function_strings(regressor_names)
    # labelstr = parse_function_str_add_dots(regressor_names)
    labelstr = enumerate(lhs_guess_strings)
    labelstr = ['  |  i:'.join([lhs, str(idx)]) for idx,lhs in labelstr]
    n = len(labelstr)

    figsize = tuple(np.array(dist_mat.shape) * 0.15 + 5)
    if figsize[0] > 70:
        figsize = tuple(np.array(figsize) * 0.15)
        labels = False
    print(f"Creating activation distance plot\nImage size:\t{figsize}")



    with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
        newcolors = mpl.cm.get_cmap('cividis_r', 4)
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               tight_layout=True,
                               figsize=figsize)
        im = ax.imshow(dist_mat, cmap=newcolors, vmin=3.5, vmax=-0.5)
        fig.colorbar(im, ax=ax, ticks=[*range(0, 4)])
        if labels:
            ax.set_yticks([*range(n)])
            ax.set_yticklabels(labelstr)
            ax.yaxis.set_tick_params(rotation=0, labelsize=7)
            ax.set_xticks([*range(n)])
            ax.set_xticklabels(labelstr)
            ax.xaxis.set_tick_params(rotation=90, labelsize=7)


@save_and_plot(filename='implicit_sols', stamp=True, plot=True)
def plot_implicit_sols(models, theta_labels,
                       show_labels=False, axislabels=True):

    sols = np.vstack(models['sol'].values)
    sols = np.array(sols).T

    theta_active = np.vstack(models['active'].values).any(axis=0)
    theta_labels = theta_labels[theta_active]
    theta_labels = d_to_dot(latexify(theta_labels))
    sols = sols[theta_active, :]

    fits = models['fit']

    lhs_labels = models['lhs'].values
    lhs_labels = latexify(lhs_labels)
    lhs_labels = d_to_dot(lhs_labels)
    indices = range(len(lhs_labels))
    fit_str = [str(np.round(fit,4)) for fit in fits]
    lhs_labels = [r' | '.join([lhs, str(fit), ':'.join(['idx',str(idx)])]) for idx,fit,lhs in zip(indices, fit_str, lhs_labels)]

    figsize = tuple(np.array(sols.shape) * 0.18 + 3)
    if figsize[1] > 70:
        figsize = np.array(figsize)
        figsize[1] = figsize[1] * 0.5
        figsize = tuple(figsize)
        axislabels = False
    if show_labels:
        figsize = tuple(np.array(figsize)+2)
    print(f"Creating implicit solutions plot\nImage size:\t{figsize}")

    fig, ax = plt.subplots(1, 1,
                           figsize=figsize, tight_layout=True)
    with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
        ax.matshow(sols.T, cmap='bwr', vmin=-0.01, vmax=0.01)
        if axislabels:
            ax.set_yticks([*range(sols.shape[1])])
            ax.set_yticklabels(lhs_labels)
            ax.yaxis.set_tick_params(rotation=0, labelsize=12)
            ax.set_xticks([*range(len(theta_labels))])
            ax.set_xticklabels(theta_labels)
            ax.xaxis.set_tick_params(rotation=90, labelsize=12)
        if show_labels:
            for (row, col), val in np.ndenumerate(sols):
                ax.text(row, col, '{:0.2f}'.format(val), ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3',
                                  alpha=0.7))

@save_and_plot(filename='ksi', stamp=True, plot=True)
def plot_ksi_fig(ksi, theta, dx, show_sparse=True, title=None):
    fig = plt.figure(tight_layout=True, figsize=(4, 8))
    if title == None:
        pass
    else:
        plt.suptitle(title)
    ax = plt.gca()
    plot_ksi(ksi, theta, dx, ax, show_sparse)
    return fig


@save_and_plot(filename='ksi_comp')
def compare_ksi(ksi1, theta1, ksi2, theta2, dx, show_sparse=True):
    fig, ax = plt.subplots(1, 2, tight_layout=True)
    plot_ksi(ksi1, theta1, dx, ax[0], show_sparse=show_sparse)
    plot_ksi(ksi2, theta2, dx, ax[1], show_sparse=show_sparse)
    return fig

@save_and_plot(filename='ksi_comp')
def plot_svd(svd):
    dims = svd['U'].shape[1]
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)

    p0 = axs[0].matshow(svd['U'], cmap='cividis')
    plt.colorbar(p0, ax=axs[0])
    axs[0].set_xticks([*range(0, dims)])
    axs[0].set_xticklabels(['PC' + str(i + 1) for i in range(dims)])
    axs[0].set_yticks([*range(0, dims)])
    axs[0].set_yticklabels(['X' + str(i + 1) for i in range(dims)])
    axs[0].set_title("Left Singular Vectors\nPrincipal Components")

    p1 = axs[1].matshow(svd['Sigma'], cmap='viridis')
    plt.colorbar(p1, ax=axs[1])
    axs[1].set_xticks([*range(0, dims)])
    axs[1].set_xticklabels([str(i + 1) for i in range(dims)])
    axs[1].set_yticks([*range(0, dims)])
    axs[1].set_yticklabels([str(i + 1) for i in range(dims)])
    axs[1].set_title("Singular Values")
    return fig

@save_and_plot(filename='lorentz3d', plot=True)
def plot_lorentz3d(state_data, title=None, **kwargs):
    with plt.style.context({'./images/BystrickyK.mplstyle'}):
        fig = plt.figure(tight_layout=True, figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.plot3D(state_data[:, 0], state_data[:, 1], state_data[:, 2], **kwargs)
        ax.scatter3D(state_data[[0, -1],0], state_data[[0, -1],1], state_data[[0, -1], 2], s=60, edgecolors='k', linewidths=2)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$x_3$")
        return ax

def plot_lorentz3d_ax(state_data, ax):
        ax.plot3D(state_data[:, 0], state_data[:, 1], state_data[:, 2])
        ax.scatter3D(state_data[[0, -1],0], state_data[[0, -1],1], state_data[[0, -1], 2], s=60, edgecolors='k', linewidths=2)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$x_3$")

# Plot analytic and spectral derivatives
@save_and_plot(filename='derivatives', plot=True)
def plot_dxdt_comparison(sig):
    dims = sig.dims
    t = sig.t
    dxdt_exact = sig.dxdt_exact.values
    dxdt_spectral = sig.dxdt_spectral.values
    if sig.kernel:
        dxdt_spectral_filtered = sig.dxdt_spectral_filtered.values
    dxdt_findiff = sig.dxdt_finitediff.values

    axmax = np.max(dxdt_exact, axis=0)
    axmin = np.min(dxdt_exact, axis=0)
    print(axmax)

    colors = color_palette('deep')
    with plt.style.context({'seaborn', './images/BystrickyK.mplstyle'}):
        fig, axs = plt.subplots(dims, 1, figsize=(12, 8), sharex='all', tight_layout=True)
        fig.suptitle('')
        plt.xlabel(r'$Time \ t \  [s]$')
        ylabels = [r'$\.{x}_{' + str(i + 1) + '} [s^{-1}]$' for i in range(sig.dims)]
        for ii, ax in enumerate(axs):
            ax.plot(t, dxdt_spectral[:, ii], '-', color=colors[1], alpha=0.2, linewidth=1, label='Spectral Cutoff')
            # ax.plot(t, dxdt_findiff[:, ii], '-', color=colors[2], alpha=0.2, linewidth=1, label='Forward Finite Difference')
            if sig.kernel:
                ax.plot(t, dxdt_spectral_filtered[:, ii], '-', color=colors[3], alpha=0.8, linewidth=1.5,
                        label='Spectral Cutoff Filtered')
            ax.plot(t, dxdt_exact[:, ii], color='black', alpha=1, linewidth=2, label='Exact')
            ax.set_ylabel(ylabels[ii])
            ax.set_xlim(xmin=-0.1, xmax=5)
            ax.set_ylim(ymin=axmin[ii]*1.1, ymax=axmax[ii]*1.1)
        axs[0].legend(loc=1, bbox_to_anchor=(1.28, 1))
    return fig

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

