def plot_optimal_diagonal():
    N = k

    fig, axes = plt.subplots(ncols=N - 1, figsize=(10, 5), sharex=True, sharey=True)
    axes = axes if hasattr(axes, '__getitem__') else [axes]
    width = 0.35
    for ax, dd in zip(axes, data):
        ax.bar(range(N), eigenvalues[:N], width, label=r'$\sigma^2 \lambda_i^{-1}$', color='b')

        k = dd['extra'].size
        ax.bar(range(k), dd['eigs'], width, color='b')
        ax.bar(range(k), dd['extra'], width, bottom=dd['eigs'], label=r'$\eta_i$', color='r')
        ax.set_ylim((0, m))

        if k > 0:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.axes.get_yaxis().set_visible(False)

    axes[0].legend()
    fig.suptitle(f"{m} observations")
    plt.tight_layout()
    plt.show()
