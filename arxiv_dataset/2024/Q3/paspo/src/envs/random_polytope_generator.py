import numpy as np
import pypoman


def generate_random_polytope(n_dim, n_points=30, seed=12345):
    def convex_hull(points):
        from scipy.spatial import ConvexHull

        """Compute the convex hull of a set of points.

        Parameters
        ----------
        points :
            Set of points.

        Returns
        -------
        :
            List of polytope vertices.
        """

        hull = ConvexHull(points, qhull_options="")
        return points[hull.vertices]

    rng = np.random.default_rng(seed)
    # generate random points
    points = rng.dirichlet(np.ones(n_dim), n_points)


    points = np.concatenate((points, np.zeros(
        (1, n_dim))))  # add zero point to have a full dimensional polytope so that we can compute the convex hull

    points = convex_hull(points)

    points = points[points.sum(axis=1) != 0.0]  # re,pve the zero point again

    A, b = pypoman.compute_polytope_halfspaces(points)
    

    return points, A, b


if __name__ == "__main__":
    generate_random_polytope(n_dim=12, n_points=30)