"""SWtools_ext_SRM2D

This module extends the functionality of SWtools_base by implementing a
spectral renormalization method for d=2.

"""
import numpy as np
import matplotlib.pyplot as plt
from .SWtools_base import IterBase
from mpl_toolkits.mplot3d import axes3d


FT = np.fft.ifftn
IFT = np.fft.fftn
FTFREQ = np.fft.fftfreq
SHIFT = np.fft.fftshift


class SRM2D(IterBase):
    """Spectral renormalization method (SRM) for d=2.

    Note
    ----
    This implementation addresses the case of a two-dimensional (d=2)
    transverse coordinate xi=(xi_1, xi_2).
    """
    def __init__(self, xi, coeffs, F, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        """Initialization of the iteration base class.

        Parameters
        ----------
        xi : tuple of d=2 arrays
            Discretized coordinates provided as tuple xi=(xi_1, xi_2), where
            xi_1 and xi_2 are d=2 coordinate arrays provided by numpys meshgrid
            with 'ij'-indexing.
        cL : array_like
            Coefficients c_L = (c_1, c_2, c_3, c_4) defining the linera operator L.
        F : function
            Nonlinear functional with call signature `F(I, xi)`, where the
            arguments are the squared magnitute of U (`I`), and the transverse
            coordinate xi (`xi`).
        tol : float, optional
            Iteration is stopped if the accuracy falls below this tolerance
            threshold (default: 1e-12).
        maxiter : int, optional
            Maximum number of allowed iterations (default: 10000).
        nskip : 1, optional
            Number of iterations to skip in between storing intermediate results (default: 1)
        verbose : bool
            Set to True to print details during iteration (default: False)
        """
        IterBase.__init__(self, xi, tol=tol, maxiter=maxiter, nskip=nskip, verbose=verbose)

        # -- SETUP AND INITIALIZATION
        xx, yy = xi
        # ... 2D VOLUME ELEMENT USED WHEN INTEGRATING
        dx, dy = xx[1,0]-xx[0,0], yy[0,1]-yy[0,0]
        self.dV = dx*dy
        # ... FOURIER SAMPLES OF THE TRANSVERSE COORDINATES
        kx = FTFREQ(xx.shape[0], d=dx)*2*np.pi
        ky = FTFREQ(yy.shape[1], d=dy)*2*np.pi
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
        # ... FOURIER REPRESENTATION OF THE LINEAR OPERATOR
        cx, cy = coeffs
        self.Lk = cx*kxx**2 + cy*kyy**2
        # ... NONLINEAR FUNCTIONAL
        self.F = F
        # ... CONVERGENCE EXPONENT USED IN FIXED POINT ITERATION
        self.gam = 1.5


    def functional_N(self, U):
        """Functional number 1 (d=2).

        Parameters
        ----------
        U : array_like
            Current solution.

        Returns
        -------
        N : float
            Value of the functional N at the current iteration step.
        """
        return np.sum(np.abs(U)**2)*self.dV

    def functional_H(self, U):
        """Functional number 2 (d=2).

        Implements functional H by employing spectral derivatives to handle the
        linear operator L.

        Parameters
        ----------
        U : array_like
            Current solution.

        Returns
        -------
        H : float
            Value of the functional H at the current iteration step.
        """
        xi, Lk, F, dV = self.xi, self.Lk, self.F, self.dV
        return np.real(np.sum( np.conj(U)*IFT(Lk*FT(U)) + F(np.abs(U)**2, xi)*np.abs(U)**2)*dV)

    def singleUpdate(self, U, N, H, kap, **kwargs):
        """Single iteration step of the d=2 SRM.

        Implements a single step of the d=2 SRM following [M2004,A2005].

        References
        ----------
        [M2004] Z. Musslimani, J. Yang, Self-trapping of light in a
        two-dimensional photonic lattice, J. Opt. Soc. Am.  B 21 (2004) 973,
        https://doi.org/10.1364/JOSAB.21.000973.

        [A2005] M. J. Ablowitz, Z. H. Musslimani, Spectral renormalization
        method for computing self-localized solutions to nonlinear systems,
        Opt. Lett. 30 (2005) 2140, https://doi.org/10.1364/OL.30.002140.

        Parameters
        ----------
        U : array_like
            Current solution.
        N : float
            Current value of functional 1.
        H : float
            Current value of functional 2.
        kap : float
            Eigenvalue of the sought-for solution.

        Returns
        -------
        U : array_like
            Updated solution.
        """
        # -- STRIP SELF KEYWORD
        xi, Lk, F, gam, dV = self.xi, self.Lk, self.F, self.gam, self.dV
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _IP = lambda f,g: np.sum(np.conj(f)*g)*dV

        # -- SRM UPDATE
        # (1) PROPOSE UPDATED SOLUTION
        U_tmp = IFT(FT(F(np.abs(U)**2,xi)*U)/(kap - Lk))
        # (2) RESCALE SOLUTION SO IT SATISFIES THE DESIRED INTEGRAL IDENTITY
        s_tmp = np.abs(_IP(U,U)/_IP(U,U_tmp))
        U_tmp *= s_tmp**gam

        return U_tmp

    def show(self, f_name='none'):
        """Prepare simple plot showing the results.

        Uses Pythons matplotlibe to prepare a two-panel figure that might be
        captioned as follows:

        Nonlinear bound state of the considered nonlinear eigenvalue problem.
        Left panel: Solution U. Right panel: Variation of the accuracy upon
        iteration.

        Parameters
        ----------
        f_name : str, optional
            Figure name. If no name is set, the figure is displayed directly.
        """
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        helper_show_d2(xi, U, it, acc, f_name=f_name)


def helper_show_d2(xi, U, it, acc, f_name='none'):
    """Helper function for making simple plots in dimension d=2.

    Uses Pythons matplotlibe to prepare a two-panel figure that might be
    captioned as follows:

    Nonlinear bound state of the considered nonlinear eigenvalue problem. Left
    panel: Solution U.  Shown are the real part (Re[U]), imaginary part
    (Im[U]), and modulus (|U|) of the solution.  Right panel: Variation of the
    accuracy upon iteration.

    Parameters
    ----------
    xi : array_like
        Discretized transverse coordinate xi.
    U : array_like
        Solution of the nonlinear eigenvalue problem.
    it : array_like
        List of iteration steps at which results where stored.
    acc : array_like
        List of accuracy values corresponding to the iteration steps in `it`.
    f_name : str, optional
        Figure name. If no name is set, the figure is displayed directly.
    """
    # -- FIGURE LAYOUT 
    fig = plt.figure(figsize=(7.,3.5))

    # -- LEFT SUBPLOT
    xx, yy = xi
    U = np.real(U)
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(xx, yy, U, edgecolor='C0', lw=0.5, rstride=5, cstride=5, alpha=0.3)
    ax.set_xlabel(r'Coordinate $\xi_1$')
    ax.set_ylabel(r'Coordinate $\xi_1$')
    ax.set_zlabel(r'Solution $U$')

    # -- RIGHT SUBPLOT
    ax = fig.add_subplot(1,2,2)
    ax.plot(it, np.log10(acc), color='C0')
    ax.set_xlabel(r"Iteration $n$")
    ax.set_ylabel(r"log-accuracy $\log(\epsilon_n)$")

    # -- SHOW/SAVE FIGURE
    plt.subplots_adjust(left=0.02, right=0.96,  bottom=0.14, top=0.96, wspace=0.5)

    if f_name=='none':
        plt.show()
    else:
        plt.savefig(f_name)

