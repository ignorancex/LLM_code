"""SWtools_base

This module implements data structures and algorithms that allow a user to
conveniently calculate solitary-wave solutions for a generalized
nonlinear-Schrödinger type equation.

"""
import os
import sys
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt

# -- FFT ABREVIATIONS
FT = nfft.ifft
IFT = nfft.fft
FTFREQ = nfft.fftfreq
SHIFT  = nfft.fftshift


class IterBase(object):
    """Base class for the implemented iteration methods.

    Attributes
    ----------
    xi : array_like
        Discretized coordinate xi.
    tol : float
        Convergence is assumed if relative error falls below this value
        (defaulf: tol=1e-12).
    maxiter : int
        Maximum number of iterations to perform (default: maxiter=10000).
    nskip : float
        Number of iterations to skip in between storing intermediate results
        (default: 1).
    verbose : bool
        Set to True to print details during iteration (default: False).
    iter_list : array_like
        Sequence of iteration steps at which intermediate results where stored
        (separated by nskip steps).
    U_list : array_like
        List of intermediate solutions U.
    acc_list : array_like
        List of intermediate accuracies (see class method _accuracy() below).
    N_list : array_like
        List of intermediate values of the integral N[U].
    H_list : array_like
        List of intermediate values of the integral H[U].
    K_list : array_like
        List of intermediate values of the eigenvalue estimate K=H[U]/N[U].

    """
    def __init__(self, xi, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        """Initialization of the iteration base class.

        Parameters
        ----------
        xi : array_like
            Discretized coordinate xi.
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
        self.xi = xi
        self.tol = tol
        self.maxiter = maxiter
        self.nskip = nskip
        self.verbose = verbose
        self.iter_list = []
        self.U_list = []
        self.acc_list = []
        self.N_list = []
        self.H_list = []
        self.K_list = []

    @property
    def U(self):
        '''float: Final solution U.'''
        return self.U_list[-1]

    @property
    def kap(self):
        '''float: Value of the eigenvalue.'''
        return self.K_list[-1]

    @property
    def N(self):
        '''float: Value of functional 1.'''
        return self.N_list[-1]

    @property
    def H(self):
        '''float: Value of functional 2.'''
        return self.H_list[-1]

    @property
    def acc(self):
        '''float: Terminal accuracy.'''
        return self.acc_list[-1]

    @property
    def num_iter(self):
        '''int: Number of iterations performed.'''
        return self.iter_list[-1]

    def _clean_up(self):
        '''Cleanup of attributes at beginning of each solution run.'''
        self.U_list = []
        self.acc_list = []
        self.iter_list = []
        self.N_list = []
        self.H_list = []
        self.K_list = []

    def _update_results(self, it, U, N, H, kap, err):
        '''Update attributes using the supplied intermediate results.'''
        self.iter_list.append(it)
        self.N_list.append(N)
        self.H_list.append(H)
        self.K_list.append(kap)
        self.U_list.append(np.copy(U))
        self.acc_list.append(err)
        if self.verbose:
            print(self)

    def solve(self, U, dum, **kwargs):
        '''Performs iteration procedure.

        Parameters
        ----------
        U : array_like
            Initial condition.
        dum : float
            Eigenvalue (in case of SRM) or value of the normalization
            constraint (in case of NSOM).
        ortho_set : array_like, keyword argument
            List of previously found orthogonal solutions.
        acc_fun : function, keyword argument
            Custom accuracy function with call signature `acc_fun(xi, Up, Uc)`,
            where the arguments are the coordinate xi (`xi`), the solution at the
            previous iteration step (`Up`), and the solution at the current
            iteration step (`Uc`).

        Returns
        -------
        U : array_like
            Final solutions.
        acc : float
            Terminal accuracy.
        succ : bool
            Boolean flag indicating if teh iteration procedure exited
            successfully.
        msg : str
            Cause of the termination.
        '''
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS
        # ... STRIP SELF KEYWORD
        xi, tol, maxiter, nskip = self.xi, self.tol, self.maxiter, self.nskip
        # ... LOCAL FUNCTIONS
        acc_fun_def = lambda xi, U, Un: np.sqrt(np.sum(np.abs(Un - U)**2)/np.sum(np.abs(U)**2))
        # -- LOGIC TO MAP OUT KEYWORD ARGUMENTS
        ortho_set = kwargs.get("ortho_set", [])
        acc_fun = kwargs.get("acc_fun", acc_fun_def)
        # -- CLEAN UP
        self._clean_up()

        # -- PREPARE/EVALUATE TRIAL SOLUTION
        U = U.astype(np.complex128)
        N = self.functional_N(U)
        H = self.functional_H(U)
        # -- INITIALIZE ITERATION SCHEME
        it = 1
        acc = np.inf
        # -- KEEP INITIAL RESULTS
        self._update_results(0, U, N, H, H/N, acc)
        # -- ITERATION PROCEDURE
        while acc > tol and it < maxiter:
            # ... ORTHOGONALIZE WRT NOT-LIST 
            if ortho_set:
                U = self.orthogonalize(U, ortho_set)
            # ... UPDATE SOLUTION 
            U_new = self.singleUpdate(U, N, H, dum, **kwargs)
            # ... ROOT-MEAN SQUARED DEVIATION W.R.T. PREVIOUS STEP 
            acc = acc_fun(xi, U, U_new)
            # ... ADVANCE UPDATE TO NEXT ITERATION STEP 
            U = U_new
            # ... RE-EVALUATE THE INTEGRALS FOR THE CURRENT SOLUTION 
            N = self.functional_N(U_new)
            H = self.functional_H(U_new)
            # ... KEEP INTERMEDIATE RESULTS
            if it%nskip==0:
                self._update_results(it, U, N, H, H/N, acc)
            it+=1

        # -- KEEP FINAL SOLUTION 
        # ... ORTHOGONALIZE WRT NOT-LIST 
        if ortho_set:
            U = self.orthogonalize(U, ortho_set)
        self._update_results(it, U, N, H, H/N, acc)

        # -- BASIC STATUS MESSAGE
        if it<maxiter:
            msg = f'Iteration terminated successfully (num_iter = {it}).'
        else:
            if it>=maxiter:
                msg = 'Maximum number of iterations reached.'
            if np.isnan(np.any(U)) or np.isnan(acc):
                msg = 'NaN result encountered.'

        if self.verbose:
            print(f"# {msg}")
            print(f"# Functional 1: N = {N}")
            print(f"# Functional 2: H = {H}")
            print(f"# Eigenvalue: K = {H/N}")
            print(f"# Local error: acc = {acc}")

        return self.U, self.acc, it<maxiter, msg

    def orthogonalize(self, U, ortho_set):
        '''Otrhogonalization procedure.

        Implements a straight forward projection construction that turns the
        solution U into a solution U_ortho, which is orthogonal to any of the
        functions in the set `ortho_set`.

        Parameters
        ----------
        U : array_like
            Current solution.
        ortho_set : array_like
            Orthogonal set of previously found solutions.

        Returns
        -------
        U_ortho : array_like
            New solution which is orthogonal to any previously found solution.
        '''
        xi = self.xi
        # -- PROJECTION CONSTRUCTION 
        for V in ortho_set:
            # ... PROJECT OUT COMPONENT V 
            U -= V*np.sum(np.conj(V)*U)/np.sum(np.abs(V)**2)
        return U

    def functional_H(self, U):
        """Functional number 2.

        Parameters
        ----------
        U : array_like
            Current solution.

        Raises
        ------
        NotImplementedError
            Must be overwritten by subclass.
        """
        raise NotImplementedError

    def functional_N(self, U):
        """Functional number 1.

        Parameters
        ----------
        U : array_like
            Current solution.

        Raises
        ------
        NotImplementedError
            Must be overwritten by subclass.
        """
        raise NotImplementedError

    def singleUpdate(self, U, N, H, dum, **kwargs):
        """Single update of the solution.

        Parameters
        ----------
        U : array_like
            Current solution.
        N : float
            Current value of functional 1.
        H : float
            Current value of functional 2.
        dum : float
            Eigenvalue (in case of SRM) or value of the normalization
            constraint (in case of NSOM).

        Raises
        ------
        NotImplementedError
            Must be overwritten by subclass.
        """
        raise NotImplementedError

    def __str__(self):
        """str: String representation of the current solution."""
        myStr = f"Iter {self.num_iter:06d}: H = {self.H:7.6F}, N = {self.N:7.6F}, K = {self.kap:7.6F}, acc = {self.acc:4.3E}"
        return myStr


class SRM(IterBase):
    """Spectral renormalization method (SRM).

    Note
    ----
    This implementation addresses the case of a one-dimensional (d=1)
    transverse coordinate xi and nonlinear functional quadratic in U.
    """
    def __init__(self, xi, cL, F, tol=1e-12, maxiter=10000, nskip=1, verbose=False, spm=False):
        """Initialization of the iteration base class.

        Parameters
        ----------
        xi : array_like
            Discretized coordinate xi.
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
        spm : bool
            Set to True to enable a structure-preserving method for a GNSE with
            nodeless, even, and real-valued solution (default: False).

            Enabling this option requires some caution: intented for  GNSEs
            with even and real solitons, it yields catastrophic results for
            GNSEs with complex-valued, oscillating-tail solitons.


        """
        IterBase.__init__(self, xi, tol=tol, maxiter=maxiter, nskip=nskip, verbose=verbose)
        self.F = F
        self.Lw = self._set_Lk(xi, cL)
        self.gam = 1.5
        self.spm = spm

    def _set_Lk(self, xi, coeffs):
        c1, c2, c3, c4 = coeffs
        k = FTFREQ(xi.size, d=xi[1]-xi[0])*2*np.pi
        return c1*k + c2*k**2 + c3*k**3 + c4*k**4

    def functional_N(self, U):
        """Functional number 1.

        Parameters
        ----------
        U : array_like
            Current solution.

        Returns
        -------
        N : float
            Value of the functional N at the current iteration step.
        """
        return np.trapz(np.abs(U)**2, x=self.xi, axis=-1)

    def functional_H(self, U):
        """Functional number 2.

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
        xi, Lw, F = self.xi, self.Lw, self.F
        return np.real(np.trapz( np.conj(U)*IFT(Lw*FT(U)) + F(np.abs(U)**2, xi)*np.abs(U)**2, x=xi, axis=-1))

    def singleUpdate(self, U, N, H, kap, **kwargs):
        """Single iteration step of the SRM.

        Implements a single step of the d=1 SRM, thoroughly detailed in
        [A2003,M2004,A2005,A2009].

        Note
        ----
        This implementation addresses the case of a one-dimensional (d=1)
        transverse coordinate xi and nonlinear functional quadratic in U.

        If the boolean flag spm=True, the method implements the
        structure-preserving add-on discussed in Appendix B of [F2008].

        References
        ----------
        [A2003] M. J. Ablowitz, Z. H. Musslimani, Discrete spatial solitons in
        a diffraction-managed nonlinear waveguide array: a unified approach,
        Physica D 184 (2003) 276–303,
        https://doi.org/10.1016/S0167-2789(03)00226-4.

        [M2004] Z. Musslimani, J. Yang, Self-trapping of light in a
        two-dimensional photonic lattice, J. Opt. Soc. Am.  B 21 (2004) 973,
        https://doi.org/10.1364/JOSAB.21.000973.

        [A2005] M. J. Ablowitz, Z. H. Musslimani, Spectral renormalization
        method for computing self-localized solutions to nonlinear systems,
        Opt. Lett. 30 (2005) 2140, https://doi.org/10.1364/OL.30.002140.

        [F2008] G. Fibich, Y. Sivan, M. I. Weinstein, Bound states of nonlinear
        Schrödinger equations with a periodic nonlinear microstructure, Physica
        D 217 (2006) 31, https://doi.org/10.1016/j.physd.2006.03.009.

        [A2009] M. Ablowitz, T. Horikis, Solitons and spectral renormalization
        methods in nonlinear optics, Eur. Phys. J.  Spec. Top. 173 (2009) 147,
        https://doi.org/10.1140/epjst/e2009-01072-0.

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
        xi, Lw, F, gam, spm = self.xi, self.Lw, self.F, self.gam, self.spm
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _IP = lambda f,g: np.trapz(np.conj(f)*g, x=xi)

        # -- SRM UPDATE
        # (1) PROPOSE UPDATED SOLUTION
        U_tmp = IFT(FT(F(np.abs(U)**2,xi)*U)/(kap - Lw))
        # (2) RESCALE SOLUTION SO IT SATISFIES THE DESIRED INTEGRAL IDENTITY
        s_tmp = np.abs(_IP(U,U)/_IP(U,U_tmp))
        U_tmp *= s_tmp**gam
        # (3) ENABLE STRUCTURE-PRESERVING ADD-ON FOR EVEN AND REAL SOLUTIONS
        if spm:
         U_tmp = np.abs(U_tmp)

        return U_tmp

    def show(self, f_name='none'):
        """Prepare simple plot showing the results.

        Uses Pythons matplotlibe to prepare a two-panel figure that might be
        captioned as follows:

        Nonlinear bound state of the considered nonlinear eigenvalue problem. Left
        panel: Solution U.  Shown are the real part (Re[U]), imaginary part
        (Im[U]), and modulus (|U|) of the solution.  Right panel: Variation of the
        accuracy upon iteration.

        Parameters
        ----------
        f_name : str, optional
            Figure name. If no name is set, the figure is displayed directly.
        """
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        helper_show_d1(xi, U, it, acc, f_name=f_name)



class NSOM(IterBase):
    """Nonlinear Successive Overrelaxation Method (NSOM).

    Note
    ----
    This implementation addresses the case of a one-dimensional (d=1)
    transverse coordinate xi.
    """
    def __init__(self, xi, cL, F, ORP=1.0, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        """Initialization of the iteration base class.

        Parameters
        ----------
        xi : array_like
            Discretized coordinate xi.
        cL : array_like
            Coefficients c_L = (c_1, c_2, c_3, c_4) defining the linera operator L.
        F : function
            Nonlinear functional with call signature `F(I, xi)`, where the
            arguments are the squared magnitute of U (`I`), and the transverse
            coordinate xi (`xi`).
        ORP : float, optional
            Overrelaxation parameter (default: 1). Needs to satisfy 0 < ORP <
            2. If ORP<1, the method works by underrelaxation.
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
        self.dxi = xi[1]-xi[0]
        self.F = F
        self.coeffs = self._set_coeffs(cL)
        self.ORP = ORP

    def _set_coeffs(self,coeffs):
        dxi = self.dxi
        dxi2 = dxi**2
        c1, c2, c3, c4 = coeffs
        return [ 1j*dxi*c1/12 + c2/12 + 0.5j*c3/dxi + c4/dxi2,\
                -2j*dxi*c1/3 -c2*4/3 - 1j*c3/dxi - c4*4/dxi2,\
                -c2*5/2 - c4*6/dxi2,\
                 2j*dxi*c1/3 -c2*4/3 + 1j*c3/dxi - c4*4/dxi2,\
                -1j*dxi*c1/12 + c2/12 - 0.5j*c3/dxi + c4/dxi2]

    def functional_N(self, U):
        """Functional number 1.

        Parameters
        ----------
        U : array_like
            Current solution.

        Returns
        -------
        N : float
            Value of the functional N at the current iteration step.
        """
        return np.trapz(np.abs(U)**2, x=self.xi)

    def functional_H(self, U):
        """Functional number 2.

        Implements functional H by employing a discretized linear operator L in
        which all derivatives are replaced by their corresponding five-point
        finite-difference approximations.

        Parameters
        ----------
        U : array_like
            Current solution.

        Returns
        -------
        H : float
            Value of the functional H at the current iteration step.
        """
        xi, dxi, F = self.xi, self.dxi, self.F
        am2, am1, a0, ap1, ap2 = self.coeffs
        HL = np.trapz( (am2*U[:-4] + am1*U[1:-3] - a0*U[2:-2] + ap1*U[3:-1] + ap2*U[4:] )*np.conj(U[2:-2]) , dx=1)/dxi
        HN = np.trapz(F(np.abs(U)**2, xi)*np.abs(U)**2, x=xi)
        return np.real(HL + HN)

    def singleUpdate(self, U, N, H, N0, **kwargs):
        """Single iteration step of the NSOM.

        Implements a single step of a nonlinear successive overrelaxation
        method [LM2019] for a generalized nonlinear Schrödinger equation. The
        procedure uses a Gauss-Seidel method with overrelaxation to iteratively
        updata a user-provided initial condition in place [NR2007,LL2017].

        References
        ----------
        [NR2007] W. Press, S. Teukolsky, W. Vetterling, B. Flannery, Numerical
        Recipes: The Art of Scientific Computing, Cambridge University Press
        (2007).

        [LL2017] H. P. Langtangen, S. Linge, Finite-Difference Computing with PDEs,
        Springer (2017).

        [LM2019] H. P. Langtangen, K.-A. Mardal, Introduction to Numerical Methods
        for Variational Problems, Springer (2019).

        Parameters
        ----------
        U : array_like
            Current solution.
        N : float
            Current value of functional 1.
        H : float
            Current value of functional 2.
        dum : float
            Eigenvalue (in case of SRM) or value of the normalization
            constraint (in case of NSOM).

        Returns
        -------
        U : array_like
            Updated solution.
        """
        # -- STRIP SELF KEYWORD
        xi, dxi, F, ORP = self.xi, self.dxi, self.F, self.ORP
        am2, am1, a0, ap1, ap2 = self.coeffs
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _norm = lambda U, N0: U*np.sqrt(N0/np.trapz(np.abs(U)**2,x=xi))

        # -- GAUSS-SEIDEL RELAXATION PROCEDURE
        for i in range(2,xi.size-2):
            # ... KEEP OLD VALUE FOR REFERENCE
            Ui = U[i]
            xii = xi[i]
            # ... AVERAGE VALUE ACCROSS LOCAL NEIGHBORHOOD
            Ui_nbs_wgt = am2*U[i-2] + am1*U[i-1] + ap1*U[i+1] + ap2*U[i+2]
            # ... LOCAL POTENTIAL
            Vi = -F(np.abs(Ui)**2, xii)
            # ... UPDATE PIVOT
            num = Ui_nbs_wgt
            den =  a0 + (H/N + Vi)*dxi**2
            Ui_new = num/den
            # ... UPDATE SOLUTION USING OVERRELAXATION 
            U[i] += ORP*(Ui_new-Ui)

        # -- SCALE SOLUTION TO DESIRED ENERGY 
        U = _norm(U,N0)

        return U

    def show(self, f_name='none'):
        """Prepare simple plot showing the results.

        Uses Pythons matplotlibe to prepare a two-panel figure that might be
        captioned as follows:

        Nonlinear bound state of the considered nonlinear eigenvalue problem. Left
        panel: Solution U.  Shown are the real part (Re[U]), imaginary part
        (Im[U]), and modulus (|U|) of the solution.  Right panel: Variation of the
        accuracy upon iteration.

        Parameters
        ----------
        f_name : str, optional
            Figure name. If no name is set, the figure is displayed directly.
        """
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        helper_show_d1(xi, U, it, acc, f_name=f_name)


def helper_show_d1(xi, U, it, acc, f_name='none'):
    """Helper function for making simple plots in dimension d=1.

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

    # -- LEFT SUBPLOT
    ax1.plot(xi, np.real(U), color='C0', dashes=[], label=r"${\rm{Re}}[U]$")
    ax1.plot(xi, np.imag(U), color='C0', dashes=[2,1], label=r"${\rm{Im}}[U]$")
    ax1.plot(xi, np.abs(U), color='gray', dashes=[1,1], label=r"$|U|$")
    ax1.set_xlabel(r"Coordinate $\xi$")
    ax1.set_ylabel(r"Solution $U$")
    legend = ax1.legend()

    # -- RIGHT SUBPLOT
    ax2.plot(it, np.log10(acc), color='C0')
    ax2.set_xlabel(r"Iteration $n$")
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_n)$")

    # -- SHOW/SAVE FIGURE
    plt.tight_layout()

    if f_name=='none':
        plt.show()
    else:
        plt.savefig(f_name)


