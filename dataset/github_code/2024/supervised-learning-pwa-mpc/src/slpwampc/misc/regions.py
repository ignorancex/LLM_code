from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import polytope as pc  # TODO import cvxopt solver so it is faster
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


class Polytope:
    """Class to represent a convex polytope defined by Ax <= b with an associated label.
    Wraps the polytope class from the polytope library."""

    @property
    def A(self) -> np.ndarray:
        return self.p.A

    @property
    def b(self) -> np.ndarray:
        return self.p.b.reshape(-1, 1)

    @property
    def is_empty(self) -> bool:
        return pc.is_empty(self.p) or self.p.chebXc is None or self.V is None

    @property
    def V(self) -> np.ndarray:
        return pc.extreme(self.p)

    def __init__(self, A: np.ndarray, b: np.ndarray, label: int | None = None) -> None:
        """Initialize the polytope.

        Parameters
        ----------
        A : np.ndarray
            The matrix A in Ax <= b.
        b : np.ndarray
            The vector b in Ax <= b.
        label : int | None
            The label associated with the polytope."""
        p = pc.Polytope(A, b)
        self.p = pc.reduce(p)
        self.label = label

    def plot(
        self,
        ax: plt.Axes | Axes3D | None = None,
        color=None,
        alpha: float = 0.5,
        lw: float = 1,
        ls: str = "--",
    ) -> None:
        """Plot the polytope in 2D or 3D."""
        if len(self.A) == 0:
            raise ValueError("Cannot plot an empty polytope.")
        if self.A.shape[1] == 2:
            if ax is None:
                fig, ax = plt.subplots()
            self.p.plot(ax=ax, color=color, alpha=alpha, linewidth=lw, linestyle=ls)
        elif self.A.shape[1] == 3:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            hull = ConvexHull(self.V)
            faces = hull.simplices
            for f in faces:
                poly = Poly3DCollection([self.V[f]], alpha=0.3, edgecolor=color)
                ax.add_collection3d(poly)
        else:
            raise ValueError("Polytope must be 2D or 3D to plot.")

    def set_label(self, labeller: Callable) -> None:
        """Set the label of the polytope by passing an interior point to the labeller function."""
        if not self.is_empty:  # don't set label for empty polytopes
            x = self.get_point()
            self.label = labeller(x)

    def get_point(self) -> np.ndarray:
        """Get a point within the polytope. Returns the Cehbyshev center, which
        is gaurenteed to be inside for a convex bounded polytope."""
        if not self.is_empty:
            return self.p.chebXc.reshape(-1, 1)
        else:
            raise ValueError("Polytope is empty. Can't get interior point")

    def check_hyperplane_intersect(self, w: np.ndarray, c: float) -> bool:
        """Check if the hyperplane defined by w^T x = c intersects the polytope.

        Parameters
        ----------
        w : np.ndarray
            The normal vector of the hyperplane.
        c : float
            The offset of the hyperplane.

        Returns
        -------
        bool
            True if the hyperplane intersects the polytope, False otherwise."""
        num_vars = self.A.shape[1]
        # Check feasibility for w^T x <= c
        A1 = np.vstack([self.A, w])
        b1 = np.vstack([self.b, np.asarray(c)])
        res1 = linprog(
            c=np.zeros(num_vars), A_ub=A1, b_ub=b1, method="highs", bounds=(None, None)
        )
        # Check feasibility for w^T x >= c
        A2 = np.vstack([self.A, -w])
        b2 = np.vstack([self.b, -np.asarray(c)])
        res2 = linprog(
            c=np.zeros(num_vars), A_ub=A2, b_ub=b2, method="highs", bounds=(None, None)
        )
        # Determine if intersection exists
        if res1.success and res2.success:
            return True  # The hyperplane intersects the polytope
        elif not res1.success and not res2.success:
            return False  # The hyperplane does not intersect the polytope
        else:
            return False  # TODO resolve why this happens
            raise RuntimeError(f"Hyperplane is parallel to boundary. WHAT DO I DO?!")

    def cut_with_hyperplanes(
        self,
        w: np.ndarray,
        c: np.ndarray,
        labeller: Callable | None = None,
        debug_plot: bool = False,
    ) -> "list[Polytope]":
        """Cut the polytope with the hyperplanes defined by w^T x = c. Returning the set of polytopes returned by the cut.
        If a labeller function is provided, the label of the resulting polytopes will be set using the labeller function.

        Parameters
        ----------
        w : np.ndarray
            The normal vectors of the hyperplanes.
        c : np.ndarray
            The offsets of the hyperplanes.
        labeller : Callable | None
            A function to set the label of the resulting polytopes. By default, None, and labels are not set.
        debug_plot : bool
            If True, the splitting process is plotted in a sequence of (blocking) figures. By default, False.

        Returns
        -------
        list[Polytope]
            The resulting polytopes."""
        polytopes: list[Polytope] = [Polytope(self.A, self.b)]
        # print(f"Cutting polytope {self.A}x <= {self.b}")
        for i in range(w.shape[0]):
            new_polytopes = []
            if debug_plot:
                fig, ax = plt.subplots()
                for p in polytopes:
                    p.plot(ax=ax)
                ax.plot(
                    [-10, 10],
                    [
                        (-w[i, 0] * -10 - c[i]) / w[i, 1],
                        (-w[i, 0] * 10 - c[i]) / w[i, 1],
                    ],
                    color="black",
                    linestyle="--",
                )
                plt.xlim(-10, 10)
                plt.ylim(-10, 10)
                plt.show()
            for idx in range(len(polytopes)):
                if polytopes[idx].check_hyperplane_intersect(w[i, :], -c[i]):
                    p1 = Polytope(
                        np.vstack([polytopes[idx].A, -w[i, :]]),
                        np.vstack([polytopes[idx].b, np.asarray(c[i])]),
                    )
                    p2 = Polytope(
                        np.vstack([polytopes[idx].A, w[i, :]]),
                        np.vstack([polytopes[idx].b, -np.asarray(c[i])]),
                    )
                    if not p1.is_empty and not p2.is_empty:
                        new_polytopes.append(p1)
                        polytopes[idx] = p2
            polytopes.extend(new_polytopes)
        if labeller is not None:
            for p in polytopes:
                p.set_label(labeller)
        return polytopes


class Partition:
    """Class to represent and manipulate a partition of the state space into regions defined by polytopes."""

    @staticmethod
    def remaining_regions(
        polytopes: "list[Polytope]", space: Polytope, debug_plot: bool = False
    ) -> "list[Polytope]":
        """Subtracts the regions from the space and returns the remaining regions in the space.

        Parameters
        ----------
        regions : list[Polytope]
            The regions in the partition.
        space : Polytope
            The space to subtract the regions from.
        debug_plot : bool
            If True, the subtraction region is plotted is plotted. By default, False.

        Returns
        -------
        list[Polytope]
            The regions with the given label."""
        r: pc.Region = space.p.diff(pc.Region([p.p for p in polytopes]))
        if debug_plot and space.A.shape[1] == 2:
            fig, ax = plt.subplots()
            if len(r) > 0:
                for p in r:
                    p.plot(ax=ax)
                else:
                    r.plot(ax=ax)
            plt.show()
        elif debug_plot and space.A.shape[1] == 3:
            fig = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection="3d")
            if len(r) > 0:
                for i, p in enumerate(r):
                    p_ = Polytope(p.A, p.b)
                    p_.plot(ax=ax, color=f"C{i}")
            else:
                p_ = Polytope(r.A, r.b)
                p_.plot(ax=ax, color="C0")
            # for i, p in enumerate(polytopes):
            #     p.plot(ax=ax, color=f"C{i}")
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_zlim(-12, 12)
            plt.show()
        elif debug_plot:
            raise ValueError("Cannot plot polytopes with dimension greater than 3.")
        remaining_polytopes = (
            [Polytope(p.A, p.b) for p in r] if len(r) > 0 else [Polytope(r.A, r.b)]
        )
        return [p for p in remaining_polytopes if not p.is_empty]
