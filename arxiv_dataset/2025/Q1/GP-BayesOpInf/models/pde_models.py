# pde_models.py
"""Partial differential equation (full-order) models used in the examples."""

__all__ = [
    "Euler",
    "CubicHeatBimodal",
]

import abc
import numpy as np
import scipy.sparse
import scipy.integrate
import scipy.interpolate
import matplotlib.colors
import matplotlib.animation
import matplotlib.pyplot as plt
from IPython.display import HTML


# Base classes ================================================================
class _BasePDE(abc.ABC):
    """Base class for partial differential equations."""

    @classmethod
    @abc.abstractmethod
    def num_variables(cls):
        return NotImplemented

    # Solving -----------------------------------------------------------------
    @abc.abstractmethod
    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (N,) ndarray
            State at time ``t``.

        Returns
        -------
        (N,) ndarray
            State derivative at time ``t``.
        """
        raise NotImplementedError

    def solve(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        method: str = "RK45",
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **kwargs: dict,
    ) -> np.ndarray:
        """Solve the model with scipy.integrate.solve_ivp().

        Parameters
        ----------
        initial_conditions : (N,) ndarray
            Initial condition to start the simulation from.
        timepoints : (k,) ndarray
            Time domain over which to solve the equations.

        The following are arguments for ``scipy.integrate.solve_ivp()``.

        method : str
            Integration strategy.
        rtol : float > 0
            Relative error tolerance.
        atol : float > 0
            Absolute error tolerance.
        kwargs : dict
            Additional arguments for ``solve_ivp()``.

        Returns
        -------
        Q : (N, k) ndarray
            Solution to the PDE over the discretized space-time domain.
        """
        return scipy.integrate.solve_ivp(
            fun=self.derivative,
            t_span=[timepoints[0], timepoints[-1]],
            y0=np.array(initial_conditions),
            method=method,
            t_eval=timepoints,
            rtol=rtol,
            atol=atol,
            **kwargs,
        ).y

    # Noise model -------------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def noise(states, noise_level=0):
        """Add noise to the PDE solution.

        Parameters
        ----------
        states : (N, k) ndarray
            Solution to the PDE over the discretized space-time domain,
            including the boundary values.
        noise_level : float
            Noise percentage to add to the solution.

        Returns
        -------
        (N, k) ndarray
            Solution array with added noise.
        """
        raise NotImplementedError


# Euler equations =============================================================
class Euler(_BasePDE):
    """Full-order solver and plotting tools for incompressible Euler equations
    for an ideal gas:

             [ rho  ]         [ rho v         ]
        d/dt [ rho v] = -d/dx [ rho v^2 + p   ]
             [ rho e]         [ (rho e + p) v ],

    where the pressure p comes form the ideal gas law

        rho e = p / (gamma - 1) + rho v^2 / 2

    with heat capacity ratio gamma = 1.4.

    Transforming to the specific volume variables (v, p, 1/rho) induces a
    purely quadratic structure.

    Parameters
    ----------
    spatial_domain : (nx,) ndarray
        One-dimensional spatial domain.
    """

    num_variables = 3
    gamma = 1.4

    def __init__(self, spatial_domain):
        """Store the spatial domain."""
        dx = np.diff(spatial_domain)
        if not np.allclose(dx, dx[0]):
            raise ValueError("nonuniform spatial domain not supported")

        self.__x = spatial_domain
        self.__dx = dx[0]
        L = self.__x[-1] - self.__x[0]
        self.__nodes = np.array([0, L / 3, 2 * L / 3, L]) + self.__x[0]

    # Properties --------------------------------------------------------------
    @property
    def x(self):
        """Spatial domain."""
        return self.__x

    @property
    def dx(self):
        """Step size for the spatial discretization."""
        return self.__dx

    # Variable transformations ------------------------------------------------
    @staticmethod
    def split(states):
        """Separate the state into individual variables."""
        return np.split(states, 3)

    @classmethod
    def lift(cls, states):
        """Lift the conservative variables to the specific volume variables,
        [rho, rho v, rho e] -> [v, p, 1/rho].

        Parameters
        ----------
        states : (3nx, k) ndarray
            Conservative variables [rho, rho v, rho e].

        Returns
        -------
        lifted_states : (3nx, k) ndarray
            Specific volume variables [v, p, 1/rho].
        """
        rho, rho_v, rho_e = cls.split(states)

        v = rho_v / rho
        p = (cls.gamma - 1) * (rho_e - 0.5 * rho * v**2)
        zeta = 1 / rho

        return np.concatenate((v, p, zeta))

    @classmethod
    def unlift(cls, lifted_states):
        """Recover the conservative variables from the specific volume
        variables, [v, p, 1/rho] -> [rho, rho v, rho e].

        Parameters
        ----------
        lifted_states : (3nx, k) ndarray
            Specific volume variables [u, p, 1/rho].

        Returns
        -------
        states : (3nx, k) ndarray
            Conservative variables [rho, rho v, rho e].
        """
        v, p, zeta = cls.split(lifted_states)

        rho = 1 / zeta
        rho_v = rho * v
        rho_e = p / (cls.gamma - 1) + 0.5 * rho * v**2

        return np.concatenate((rho, rho_v, rho_e))

    @classmethod
    def lift_ddts(cls, states, ddts):
        """Lift the native state time derivatives to the time derivatives
        of the learning variables.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        ddts : (n, k) ndarray
            Time derivatives of the native state variables. Each column
            ``ddts[:, j]`` corresponds to the state vector ``states[:, j]``.
        """
        rho, rho_v, _ = cls.split(states)
        drho, drho_v, drho_e = cls.split(ddts)
        v = rho_v / rho

        dv = (drho_v - drho * v) / rho
        dp = (cls.gamma - 1) * (drho_e - rho_v * dv - drho * v**2 / 2)
        dxi = -drho / rho**2

        return np.concatenate((dv, dp, dxi))

    # Initial conditions ------------------------------------------------------
    def initial_conditions(self, init_params, plot=False):
        r"""Generate initial conditions by evaluating periodic cubic splines
        for density :math:`\rho` and velocity :math:`v`.

        Parameters
        ----------
        init_params : (6,) ndarray
            Degrees of freedom for the initial conditions, three interpolation
            values for the density and three for the velocity (in that order).
        plot : bool
            If True, display the initial conditions over the spatial domain.

        Returns
        -------
        init : (3nx,) ndarray
            Initial conditions in the LEARNING VARIABLES, [v, p, 1/rho].
        """
        # Unpack initial condition parameters and make them periodic.
        rho0s, v0s = init_params[:3], init_params[3:]
        v0s = np.concatenate((v0s, [v0s[0]]))
        rho0s = np.concatenate((rho0s, [rho0s[0]]))

        # Initial condition for velocity.
        v_spline = scipy.interpolate.CubicSpline(
            self.__nodes,
            v0s,
            bc_type="periodic",
        )
        v = v_spline(self.x)

        # Initial condition for pressure.
        p = 1e5 * np.ones_like(v)

        # Initial condition for density.
        rho_spline = scipy.interpolate.CubicSpline(
            self.__nodes,
            rho0s,
            bc_type="periodic",
        )
        rho = rho_spline(self.x)

        # Group the initial conditions together and plot if desired.
        init = np.concatenate((v, p, 1 / rho))
        if plot:
            _, axes = self.plot_space(init)
            axes[0].set_title("Initial conditions")
            axes[0].plot(self.__nodes, v0s, "k*", mew=0)
            axes[2].plot(self.__nodes, rho0s, "k*", mew=0)

        return init

    # Solving  ----------------------------------------------------------------
    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (N,) ndarray
            State at time t *IN THE CONSERVATIVE VARIABLES*.

        Returns
        -------
        (N,) ndarray
            State derivative at time t.
        """
        rho, rho_u, rho_e = self.split(state)
        u, p, _ = self.split(self.lift(state))

        def _ddx(var):
            return (var - np.roll(var, 1, axis=0)) / self.dx

        return -np.concatenate(
            [
                _ddx(rho_u),
                _ddx(rho * u**2 + p),
                _ddx((rho_e + p) * u),
            ]
        )

    def solve(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
    ) -> np.ndarray:
        """Solve the model with 4-5th order explicit Runge-Kutta time stepping.
        A cubic spline interpolation is used to evaluate the solution
        at the given time points.

        Parameters
        ----------
        initial_conditions : (N,) ndarray
            Initial condition to start the simulation from,
            given in the specific volume variables.
        timepoints : (k,) ndarray
            Time domain over which to evaluate the solution.

        Returns
        -------
        states : (N, k) ndarray
            Solution to the PDE over the discretized space-time domain,
            given in the specific volume variables.
        """
        return self.lift(
            _BasePDE.solve(
                self,
                initial_conditions=self.unlift(initial_conditions),
                timepoints=timepoints,
                method="RK45",
            )
        )

    # Noise model -------------------------------------------------------------
    def noise(self, states, noise_level=0):
        """Noise to the solution by transforming to the conservative variables,
        adding Gaussian noise (except at the initial condition), and
        transforming back to the specific volume variables.

        Parameters
        ----------
        states : (3nx, k) ndarray
            Solution to the PDE over the discretized space-time domain in terms
            of the specific volume variables.
        noise_level : float
            Noise percentage to add to the solution (Gaussian).

        Returns
        -------
        (3nx, k) ndarray
            Solution array with added noise, also in the specific volume
            variables.
        """
        if not noise_level:
            return states

        # Noise standard deviation relative to variable scale.
        unlifted = self.unlift(states[:, 1:])
        scale = np.vstack(
            [
                np.full_like(var, noise_level * (var.max() - var.min()))
                for var in self.split(unlifted)
            ]
        )

        noised = np.random.normal(
            loc=unlifted,
            # scale=unlifted * noise_level,
            scale=scale,
            size=unlifted.shape,
        )
        return np.column_stack([states[:, 0], self.lift(noised)])

    # Visualization -----------------------------------------------------------
    @staticmethod
    def _format_subplots(fig, axes):
        axes[0].set_ylabel(r"$v(x)$")
        axes[1].set_ylabel(r"$p(x)$")
        axes[2].set_ylabel(r"$\rho(x)$")
        fig.align_ylabels(axes)

    def _format_spatial_subplots(self, fig, axes):
        """Put labels on subplots for variables in space."""
        self._format_subplots(fig, axes)
        axes[2].set_xlabel(r"$x\in[0,L)$")
        axes[2].set_xlim(self.x[0], self.x[-1])

    @classmethod
    def _format_temporal_subplots(cls, fig, axes, t):
        """Put labels on subplots for variables in time."""
        cls._format_subplots(fig, axes)
        axes[2].set_xlabel(r"$t\in[t_{0},t_{f}]$")
        axes[2].set_xlim(t[0], t[-1])

    def plot_space(self, vpzeta):
        """Plot velocity, pressure, and density over space at a fixed point
        in time.

        Parameters
        ----------
        vpzeta: (3n,) ndarray
            velocity, pressure, and 1/density in a single array.

        Returns
        -------
        Figure handle, Axes handles
        """
        u, p, zeta = self.split(vpzeta)

        fig, axes = plt.subplots(3, 1, sharex=True)  # , figsize=(6, 6))
        axes[0].plot(self.x, u)
        axes[1].plot(self.x, p)
        axes[2].plot(self.x, 1 / zeta)
        self._format_spatial_subplots(fig, axes)

        return fig, axes

    def plot_time(self, t, v_p_or_zeta):
        """Plot velocity, pressure, or density in time at a fixed point in
        space.

        Parameters
        ----------
        t : (k,) ndarray
            Time domain.
        v_p_or_zeta: (k,) ndarray
            velocity, pressure, or 1/density at a single point.

        Returns
        -------
        Figure handle, Axes handles
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(t, v_p_or_zeta)
        ax.set_xlabel(r"$t\in[t_{0},t_{f}]$")
        ax.set_xlim(t[0], t[-1])
        return fig, ax

    def plot_traces(self, t, vpzeta, nlocs=20, cmap=None, isdata=False):
        """Plot traces in time at ``nlocs`` locations."""
        xlocs = np.linspace(0, vpzeta.shape[0] // 3, nlocs + 1, dtype=int)[:-1]
        xlocs += xlocs[1] // 2
        if cmap is None:
            cmap = plt.cm.twilight
        colors = cmap(np.linspace(0, 1, nlocs + 1)[:-1])
        u, p, zeta = self.split(vpzeta)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
        ls = "." if isdata else "-"
        for j, c in zip(xlocs, colors):
            axes[0].plot(t, u[j], ls, color=c, lw=1)
            axes[1].plot(t, p[j], ls, color=c, lw=1)
            axes[2].plot(t, 1 / zeta[j], ls, color=c, lw=1)
        self._format_temporal_subplots(fig, axes, t)

        # Colorbar.
        lsc = cmap(np.linspace(0, 1, 400))
        scale = matplotlib.colors.Normalize(vmin=0, vmax=1)
        lscmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "euler", lsc, N=nlocs
        )
        mappable = plt.cm.ScalarMappable(norm=scale, cmap=lscmap)
        cbar = fig.colorbar(mappable, ax=axes, pad=0.015)
        cbar.set_ticks(self.x[xlocs] / (self.x[-1] - self.x[0]))
        cbar.set_ticklabels([f"{x:.2f}" for x in self.x[xlocs]])
        cbar.set_label(r"spatial coordinate $x$", fontsize="x-large")

        return fig, axes

    def plot_spacetime(self, t, vpzeta):
        """Plot learning variables over space-time.

        Parameters
        ----------
        t : (k,) ndarray
            Time domain.
        vpzeta : (3n, k) ndarray
            The data to plot (learning variables).
        """
        if vpzeta.ndim != 2:
            raise ValueError("argument 'vpzeta' must be two dimensional")

        u, p, zeta = self.split(vpzeta)
        rho = 1 / zeta
        X, T = np.meshgrid(self.x, t, indexing="ij")

        # Plot variables in space and time.
        fig, axes = plt.subplots(
            3, 1, sharex=True, sharey=True, figsize=(6, 6)
        )
        for v, ax in zip([u, p, rho], axes):
            cdata = ax.pcolormesh(X, T, v, shading="nearest", cmap="viridis")
            fig.colorbar(cdata, ax=ax, extend="both")
            ax.set_ylabel(r"$t\in[t_{0},t_{f}]$")

        axes[-1].set_xlabel(r"$x\in[0,L)$")
        axes[0].set_title("Velocity")
        axes[1].set_title("Pressure")
        axes[2].set_title("Density")

        return fig, axes

    def animate(self, profile, skip=20):
        """Animate a single evolution profile in time.

        Parameters
        ----------
        profile : (3n,k) ndarray
            In lifted variables...
        skip : int
            Animate every `skip` snapshots, so the total number of
            frames is `k//skip`
        """
        # Process the input.
        profile = np.array(profile)
        if profile.ndim != 2:
            raise ValueError("two-dimensional data required for animation")
        data = np.split(profile, 3, axis=0)

        # Initialize the figure and subplots.
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6), dpi=150)
        lines = [ax.plot([], [])[0] for ax in axes]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            for ax, line, var in zip(axes, lines, data):
                line.set_data(self.x, var[:, index * skip])
            axes[0].set_title(rf"$t = t_{{{index*skip}}}$")
            return lines

        for ax, var in zip(axes, data):
            ax.set_ylim(var.min() * 0.95, var.max() * 1.05)
        self._format_spatial_subplots(fig, axes)
        axes[0].set_title(r"$t = t_{0}$")

        a = matplotlib.animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=profile.shape[1] // skip,
            interval=30,
            blit=True,
        )
        plt.close(fig)
        return HTML(a.to_jshtml())


# Heat equation ===============================================================
class HeatBimodal(_BasePDE):
    """Full-order solver and plotting tools for forced one-dimensional heat
    equations with constant Dirichlet boundary conditions:

        q_t = diffusion * q_xx + f(x, t)
        q(0, t) = left_bc,   q(L, t) = right_bc.

    The solver uses second-order finite differences to discretize space.

    Parameters
    ----------
    spatial_domain : (N + 2,)
        Spatial domain for all simulations, including the boundary points.
        Only uniformly spaced meshes are currently supported.
    left_bc : float
        Constant dirichlet boundary condition at x = spatial_domain[0].
    right_bc : float
        Constant dirichlet boundary condition at x = spatial_domain[-1].
    diffusion : float
        Diffusion constant.
    """

    num_variables = 1

    # Initialization ----------------------------------------------------------
    def __init__(
        self,
        spatial_domain: np.ndarray,
        left_bc: float,
        right_bc: float,
        diffusion: float = 1e-2,
        a: float = 1,
        b: float = 1,
    ):
        """Initialize the spatial domain and set up solver matrices."""
        self.__left = left_bc
        self.__right = right_bc
        self.__kappa = diffusion

        # Set the spatial domain.
        dx = np.diff(spatial_domain)
        if not np.allclose(dx, dx[0]):
            raise ValueError("nonuniform spatial domain not supported")
        self.__x = spatial_domain
        self.__dx = dx[0]

        # Construct matrices used by the full-order solver.
        self.__N = dof = spatial_domain.size - 2
        dx2inv = diffusion / dx[0] ** 2
        diags = np.array([1, -2, 1]) * dx2inv
        self.__A = scipy.sparse.diags(diags, [-1, 0, 1], (dof, dof)).tocsc()

        # Constant vector for Dirichlet BCs.
        c = np.zeros(dof, dtype=float)
        c[0] = self.left_bc * dx2inv
        c[-1] = self.right_bc * dx2inv
        self.__c = c

        # Shift function for removing nonhomogeneous boundary conditions.
        L = spatial_domain[-1] - spatial_domain[0]
        slope = (right_bc - left_bc) / L
        self._shift = left_bc + slope * (spatial_domain - spatial_domain[0])

        # Precompute some things for the forcing function.
        _fterm1 = 1 / (1 + 100 * (self.x - 0.25) ** 2)
        _fterm2 = 1 / (1 + 100 * (self.x - 0.75) ** 2)
        self.__B = np.column_stack((_fterm1, _fterm2))

        self.__params = (a, b)

    # Properties --------------------------------------------------------------
    @property
    def left_bc(self):
        """Constant value of the left boundary condition, q(x[0], t)."""
        return self.__left

    @property
    def right_bc(self):
        """Constant value of the right boundary condition, q(x[-1], t)."""
        return self.__right

    @property
    def diffusion(self):
        """Diffusion coefficient."""
        return self.__kappa

    @property
    def spatial_domain(self):
        """Spatial domain, including the boundary points."""
        return self.__x

    @property
    def x(self):
        """Spatial domain, NOT including the boundary points."""
        return self.spatial_domain[1:-1]

    @property
    def N(self):
        """Degrees of freedom: len(x) = len(spatial_domain) - 2."""
        return self.__N

    @property
    def dx(self):
        """Spatial resolution."""
        return self.__dx

    @property
    def constant(self):
        """Constant term accounting for fixed boundary conditions."""
        return self.__c

    @property
    def stiffness(self):
        """Stiffness matrix."""
        return self.__A

    @property
    def input_matrix(self):
        """Input matrix."""
        return self.__B

    @property
    def forcing_args(self):
        """Parameters for the forcing function."""
        return self.__params

    # Forcing term ------------------------------------------------------------
    @staticmethod
    def oscillators(t, a, b):
        return np.array(
            [
                a * np.sin(2 * np.pi * t),
                b * np.sin(4 * np.pi * t),
            ]
        )

    def forcing(self, t):
        """Periodic bimodal forcing function.

            f(x, t) = sin(2πt) / (1 + 100(x - 1/4)^2)
                    + sin(4πt) / (1 + 100(x - 3/4)^2)

        Parameters
        ----------
        t : float or (k,) ndarray
            Time, either an instance or array.

        Returns
        -------
        (N,) or (N, k) ndarray
            Value of the forcing function over the spatial domain.
        """
        a, b = self.forcing_args
        return self.input_matrix @ self.oscillators(t, a, b)

    # Auxiliary conditions ----------------------------------------------------
    @staticmethod
    def initial_conditions(x, alpha, beta):
        """Generate the intial conditions

        q(x, t=0) =
        x(L - x)(6 e^-x (L - x)^2 - 10 e^x sin(x / 6L)) + a + (b - a)x / L.

        When a = 0, b = 1, and L = 1, this is

        q(x, t=0) = x(1 - x)(6 e^-x (1 - x)^2 - 10 e^x sin(x / 6)) + x.

        Parameters
        ----------
        x : (N+2,) ndarray
            Spatial domain [0, ..., L] (including boundary points).
        alpha : float
            Left Dirichlet boundary condition
        beta : float
            Right Dirichlet boundary condition
        """
        L = x[-1] - x[0]
        homogeneous1 = 6 * np.exp(-x) * x * (L - x) ** 3
        homogeneous2 = 10 * np.exp(x) * x * (L - x) * np.sin(x / (L * 6))
        nonhomogeneous = alpha + (beta - alpha) / L * (x - x[0])
        return homogeneous1 - homogeneous2 + nonhomogeneous

    # Differential equation ---------------------------------------------------
    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (num_variables,) ndarray
            State at time t.

        Returns
        -------
        (num_variables,) ndarray
            State derivative at time t.
        """
        return self.constant + self.stiffness @ state + self.forcing(t)

    def jacobian(self, t, state) -> np.ndarray:
        """Calculate the Jacobian, which is always the stiffness matrix."""
        return self.stiffness

    def solve(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> np.ndarray:
        """Solve the model with scipy.integrate.solve_ivp(method="BDF").

        Parameters
        ----------
        initial_conditions : (N,) or (N+2,) ndarray
            Initial condition to start the simulation from.
            May or may not include the boundary points.
        timepoints : (k,) ndarray
            Time domain over which to solve the equations.

        The following parameters are arguments for scipy.integrate.solve_ivp().

        method : str
            Integration strategy.
        rtol : float > 0
            Relative error tolerance.
        atol : float > 0
            Absolute error tolerance.

        Returns
        -------
        Q : (N+2, k) ndarray
            Solution to the PDE over the discretized space-time domain,
            including the boundary values.
        """
        # Validate / process initial conditions.
        if initial_conditions.shape == (self.N + 2,):
            if initial_conditions[0] != self.left_bc:
                raise ValueError(
                    "initial_conditions inconsistent with left boundary "
                    f"condition q(x[0], t) = {self.left_bc:.2e}"
                )
            if initial_conditions[-1] != self.right_bc:
                raise ValueError(
                    "initial_conditions inconsistent with right boundary "
                    f"condition q(x[-1], t) = {self.right_bc:.2e}"
                )
            initial_conditions = initial_conditions[1:-1]
        if initial_conditions.shape != (self.N,):
            raise ValueError(
                f"invalid initial_conditions shape, should be ({self.N},)"
            )

        # Solve for the non-boundary DOFs.
        solution = _BasePDE.solve(
            self,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
            method="BDF",
            rtol=rtol,
            atol=atol,
            jac=self.jacobian,
        )

        # Append the boundary values to the ends.
        leftbc = np.full_like(timepoints, self.left_bc)
        rightbc = np.full_like(timepoints, self.right_bc)
        return np.vstack((leftbc, solution, rightbc))

    # Noise model -------------------------------------------------------------
    @staticmethod
    def noise(states, noise_level=0):
        """Add noise to the PDE solution, except at the initial and boundary
        conditions.

        Parameters
        ----------
        states : (N+2, k) ndarray
            Solution to the PDE over the discretized space-time domain,
            including the boundary values.
        noise_level : float
            Noise percentage to add to the solution.

        Returns
        -------
        (N+2, k) ndarray
            Solution array with added noise.
        """
        if not noise_level:
            return states

        interior = states[1:-1, 1:]

        # Add standard normal noise (except to auxiliary conditions).
        newinterior = np.random.normal(
            loc=interior,
            scale=(noise_level * interior),
            size=interior.shape,
        )
        return np.column_stack(
            [
                states[:, 0],  # Initial condition
                np.vstack([states[0, 1:], newinterior, states[-1, 1:]]),
            ]
        )

    # Visualization -----------------------------------------------------------
    def plot_space(self, state, ax=None):
        """Plot the state q(t=fixed, x) over the spatial domain.

        Parameters
        ----------
        state : (N+2,) or (N,) or (npoints, N+2) or (npoints, N) ndarray
            State variable at one or more points in time. If boundary points
            are not included, pad the state with the boundary conditions.
        ax : matplotlib.Axes
            Axes to draw on. If not provided, a new figure is created.

        Returns
        -------
        ax : matplotlib.Axes
            Axes drawn on.
        """
        state = np.atleast_2d(state)
        if state.shape[-1] == self.N:
            leftbc = np.atleast_2d(np.full_like(state[0], self.left_bc)).T
            rightbc = np.atleast_2d(np.full_like(state[0], self.right_bc)).T
            state = np.hstack((leftbc, state, rightbc))

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 3))

        for statevar in state:
            ax.plot(self.spatial_domain, statevar)
        ax.set_xlim(self.spatial_domain[0], self.spatial_domain[-1])
        ax.set_xlabel(r"$x$")

        return ax

    def plot_time(self, timepoints, state, ax=None):
        """Plot the state q(t, x=fixed) over time.

        Parameters
        ----------
        timepoints : (k)
            Time domain over which to plot the state.
        state : (k,) or (npoints, k) ndarray
            State variable at one or more points in space.
        ax : matplotlib.Axes
            Axes to draw on. If not provided, a new figure is created.

        Returns
        -------
        ax : matplotlib.Axes
            Axes drawn on.
        """
        state = np.atleast_2d(state)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 3))

        for stateloc in state:
            ax.plot(timepoints, stateloc)
        ax.set_xlim(timepoints[0], timepoints[-1])
        ax.set_xlabel(r"$t$")

        return ax

    def plot_spacetime(self, timepoints, states, nlines=6):
        """Plot the state over a space-time domain in a heat map.

        Parameters
        ----------
        timepoints : (k)
            Time domain over which to plot the state.
        states : (N, k) or (N+2, k) ndarray
            State snapshots. If boundary points are not included,
            pad the state with the boundary conditions.
        nlines : int
            Number of moments in time to draw the full state solution.

        Returns
        -------
        list of 2 maplotlib.Axes
            Axes drawn on.
        """
        # Check dimensions and pad states with BCs if needed.
        if states.ndim != 2:
            raise ValueError("states must be two-dimensional")
        if states.shape[0] == self.N:
            leftbc = np.full_like(timepoints, self.left_bc)
            rightbc = np.full_like(timepoints, self.right_bc)
            states = np.vstack((leftbc, states, rightbc))
        if timepoints.ndim > 1 or states.shape != (
            self.N + 2,
            timepoints.size,
        ):
            raise ValueError("timepoints and states not aligned")

        X, T = np.meshgrid(self.spatial_domain, timepoints, indexing="ij")
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 2))

        # Plot temperature at `nlocs` equally spaced moments in time.
        color = iter(plt.cm.viridis(np.linspace(0.25, 1, nlines)))
        for j in np.linspace(0, timepoints.size - 1, nlines).astype(int):
            ax1.plot(
                self.spatial_domain,
                states[:, j],
                color=next(color),
                label=rf"$q(x, t_{{{j}}})$",
            )
        ax1.set_xlim(self.spatial_domain[0], self.spatial_domain[-1])
        ax1.set_xlabel(r"$x$")

        # Plot temperature in space and time.
        cdata = ax2.pcolormesh(X, T, states, shading="nearest", cmap="magma")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$t$")
        fig.colorbar(cdata, ax=ax2, extend="both")

        # Make a legend on the left side of the plots.
        fig.subplots_adjust(left=0.2, wspace=0.15)
        ax1.legend(
            loc="center left",
            edgecolor="none",
            frameon=False,
            bbox_to_anchor=(0, 0.5),
            bbox_transform=fig.transFigure,
        )

        title = r"Temperature $q(x, t)$"
        fig.suptitle(title)

        return fig, [ax1, ax2]

    def animate(self, profiles, labels=None, titles=True, saveas=None):
        """Animate one or two state profiles in time.
        This method is for use in a Jupyter notebook.

        Parameters
        ----------
        profiles : (N+2, k) ndarray or two of these
            State trajectories to animate.
        labels : list of strings
            Labels for each trajectory if two are provided.
        saveas : str
            Save the animation to the indicated file.
        """
        # Pre-process data profiles.
        profiles = np.array(profiles)
        if profiles.ndim == 1:
            raise ValueError("two-dimensional data required for animation")
        if profiles.ndim == 2:
            profiles = np.array([profiles])

        # Pre-process legend labels.
        draw_legend = labels is not None
        if not draw_legend:
            labels = [None] * len(profiles)
        assert len(profiles) == len(labels)

        # Create the figure and draw blank lines.
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
        lines = [plt.plot([], [], lw=2, label=label)[0] for label in labels]

        def update(index):
            """Update the lines with data at the given time index."""
            for line, states in zip(lines, profiles):
                line.set_data(self.spatial_domain, states[:, index])
            if titles:
                ax.set_title(rf"$t = t_{{{index}}}$")
            return lines

        # Format axes and (optionally) add a legend.
        ax.set_xlim(self.spatial_domain[0], self.spatial_domain[-1])
        ax.set_ylim(profiles.min() - 0.2, profiles.max() + 0.2)
        # ax.set_xlabel(r"$x$")  # Cuts off awkwardly when saved
        if titles:
            ax.set_title(r"$t = t_{0}$")
        if draw_legend:
            fig.subplots_adjust(left=0.25)
            ax.legend(
                loc="center left",
                edgecolor="none",
                frameon=False,
                bbox_to_anchor=(0, 0.5),
                bbox_transform=fig.transFigure,
            )
        else:
            ax.set_ylabel(r"$q(x, t)$")

        # Make the animation.
        ani = matplotlib.animation.FuncAnimation(
            fig,
            update,
            frames=profiles[0].shape[1],
            interval=5,
            blit=True,
        )
        plt.close(fig)

        # Export the animation to disk if desired.
        if saveas:
            ani.save(saveas)

        # Export in a format that can be displayed in a notebook.
        return HTML(ani.to_jshtml())


class CubicHeatBimodal(HeatBimodal):
    """Full-order solver and plotting tools for a cubic, forced,
    one-dimensional heat equations with constant Dirichlet boundary conditions.

        q_t = diffusion * q_xx - q^3 + f(x, t)
        q(0, t) = left_bc,   q(L, t) = right_bc.

    The forcing function is given by

        f(x, t) = a * sin(2πt) / (1 + 100(x - 1/4)^2)
                + b * sin(4πt) / (1 + 100(x - 3/4)^2),

    where a and b are given constants (defaulting to 1).
    The solver uses second-order finite differences to discretize space.

    Parameters
    ----------
    spatial_domain : (N + 2,)
        Spatial domain for all simulations, including the boundary points.
        Only uniformly spaced meshes are currently supported.
    N : int
        Degrees of freedom. Because the Dirichlet boundary conditions are
        specified, there are len(spatial_domain) - 2 total degrees of freedom.
    dx : float
        Spatial resolution (distance between consecutive spatial points).
    left_bc : float
        Constant dirichlet boundary condition at x = spatial_domain[0].
    right_bc : float
        Constant dirichlet boundary condition at x = spatial_domain[-1].
    """

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (num_variables,) ndarray
            State at time t.

        Returns
        -------
        (num_variables,) ndarray
            State derivative at time t.
        """
        return HeatBimodal.derivative(self, t, state) - state**3

    def jacobian(self, t: float, state: np.ndarray) -> np.ndarray:
        """Calculate the derivative Jacobian."""
        return self.stiffness - np.diag(3 * state**2)
