import numpy as np
import matplotlib.pyplot as plt


def plot_ksi(ksi, theta, dx, ax, show_sparse=True, show_sparse_tol=0.1):
    ksi = ksi.T  # fix the ksi output shape
    if show_sparse:
        idx_dim_active = np.apply_along_axis(lambda x: sum(abs(x)) > show_sparse_tol, 1, ksi)
        ksi = ksi[idx_dim_active, :]
        theta = theta.iloc[:, idx_dim_active]

    p = ax.matshow(ksi, cmap='seismic', vmin=-2, vmax=2)
    plt.colorbar(p, ax=ax)
    ax.set_yticks([*range(min(theta.shape))])
    ax.set_yticklabels(theta.columns)
    ax.yaxis.set_tick_params(rotation=30)
    ax.set_ylabel("Candidate functions")
    ax.set_xticks([*range(min(dx.shape))])
    ax.set_xticklabels(dx.columns)
    ax.set_xlabel("dx/dt")
    for (col, row), val in np.ndenumerate(ksi):
        ax.text(row, col, '{:0.2f}'.format(val), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))


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


# Plot analytic and spectral derivatives
def plot_dxdt_comparison(sig):
    dims = sig.dims
    t = sig.t
    dxdt_exact = sig.dxdt_exact.values
    dxdt_spectral = sig.dxdt_spectral.values
    if sig.kernel:
        dxdt_spectral_filtered = sig.dxdt_spectral_filtered.values
    dxdt_findiff = sig.dxdt_finitediff.values

    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(dims, 1, sharex='all', tight_layout=True)
        plt.xlabel('Time t [s]')
        ylabels = [r'$\.{X}_{' + str(i + 1) + '} [s^{-1}]$' for i in range(sig.dims)]
        for ii, ax in enumerate(axs):
            ax.plot(t, dxdt_exact[:, ii], 'k', alpha=1, linewidth=2, label='Exact')
            ax.plot(t, dxdt_spectral[:, ii], '-', color='blue', alpha=0.8, linewidth=2, label='Spectral Cutoff')
            ax.plot(t, dxdt_findiff[:, ii], '-', color='c', alpha=0.5, label='Forward Finite Difference')
            if sig.kernel:
                ax.plot(t, dxdt_spectral_filtered[:, ii], '-', color='red', alpha=0.8, linewidth=2,
                        label='Spectral Cutoff Filtered')
            ax.set_ylabel(ylabels[ii])
            ax.legend(loc=1)
