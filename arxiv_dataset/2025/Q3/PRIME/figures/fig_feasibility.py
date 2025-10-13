import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def fig_feasibility():
    # make data
    u = np.linspace(-2.5, 1, 300)

    def f(x):
        return 2 * x**2 + x**3

    def f_grad(x):
        return 4 * x + 3 * x**2

    y = f(u)

    left = f(u) + f_grad(u) * (-2 - u) <= 1
    right = f(u) + f_grad(u) * (0 - u) <= 1
    feasible = left | right

    z = -1.31
    g1 = f(z) + f_grad(z) * (u - z)
    print("intersect", (1 - f(z)) / f_grad(z) + z)
    print("intersect", (0.5 - f(z)) / f_grad(z) + z)

    # override
    plt.rcParams.update({"figure.figsize": (6, 3)})

    # plot
    fig, ax = plt.subplots(layout="constrained", sharex=True)

    ax.plot(u, y, color="C0", label=r"$y = h(u)$")
    ax.plot(u, g1, color="C1", label=r"$h(-1.31)$ linearization")
    ax.axvline(x=z, color="C2", label="measurement location")

    # constraints
    ax.axvline(x=0, color="C3", linestyle="--", label="constraints")
    ax.axvline(x=-2, color="C3", linestyle="--")
    ax.axhline(y=1, color="C3", linestyle="--")
    square = Rectangle(
        label="constraint satisfying region",
        xy=(-2, 0),
        width=2,
        height=1,
        color="lightgrey",
        alpha=0.5,
    )
    ax.add_patch(square)

    # linearization
    # ax.fill_between(
    #     u, 1.5, where=feasible, facecolor="lightgray", alpha=0.5, label="feasible region"
    # )

    ax.set(
        xlim=(-2.5, 1),
        xticks=np.arange(-2.5, 1.5, 0.5),
        ylim=(0, 1.5),
        yticks=np.arange(0, 2, 0.5),
    )
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="lower right")


if __name__ == "__main__":
    fig_feasibility()
    plt.show()
