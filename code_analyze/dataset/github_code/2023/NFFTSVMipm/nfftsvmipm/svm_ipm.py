"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from scipy.linalg import lu_factor

from .kernel_matvec import kernel_matvec
from .hessian_ipm import hessian_ipm_pd
from .precond import SMW_prec
  
##################################################################################

# define function that evaluates F in Newton system at iterates alpha, lambda, xi, eta
def evaluate_F_Newton(a, b, MATa, ytrain, C, mu):
    """
    Compute the current right-hand side in the primal-dual Newton system, see equation (5) in paper.
            
    Parameters
    ----------
    a : ndarray
        Current alpha vector.
    b : float
        Current lambda value.
    MATa : ndarray
        Product of YKY with the current alpha vector.
    ytrain : ndarray
        The training target vector.
    C : float
        The regularization parameter controlling the amount of misclassification.
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    mu : float
        The barrier parameter mu.

    Returns
    -------
    F : ndarray
        Current right-hand side in the primal-dual Newton system.
    """
    n = len(a)
    e = np.ones(n)
    F1 = MATa - e + b*ytrain - (mu*e)/a + (mu*e)/(C*e-a)
    F2 = np.dot(ytrain,a)
    
    F = np.hstack((F1,F2))
    
    return F

##################################################################################
    
def svm_ipm_pd_line_search(KER, ytrain, C, iter_ip, tol, sigma_br, Gmaxiter, Gtol, prec, Ldec=[]):
    """
    Interior point method for training a support vector machine.

    Parameters
    ----------
    KER : LinearOperator
        The linear operator for multiplying the kernel matrix by a vector.
    ytrain : ndarray
        The training target vector.
    C : float
        The regularization parameter controlling the amount of misclassification.
        The relative weight of error vs. margin, such that 0 <= alpha <= C.
    iter_ip : int
        The maximum number of interior point iterations.
    tol : float
        The IPM convergence tolerance.
    sigma_br : float
        The barrier reduction parameter.
    Gmaxiter : int
        The maximum number of GMRES iterations.
    Gtol : float
        The GMRES convergence tolerance.
    prec : str
        The preconditioner that shall be used for the GMRES within the IPM.
    Ldec : ndarray, default=[]
        The low-rank decomposition matrix for the preconditioner.

    Returns
    -------
    alpha : ndarray
        The learned classifier parameter.
    GMRESiter : list
        List of the GMRES iterations at each IPM step.
    i : int
        Number of IPM iterations.
    """
    # Set parameters
    gap = 0.99995
    n = len(ytrain)
    GMRESiter = []

    # Initialize alpha: components of alpha corresponding to y=1 equal to
    # C*(1-p), components corresponding to y=-1 equal to C*p, with p fraction
    # of components with y=1.
    p = np.sum(ytrain>0)/n
    alpha = np.tile(C*p, (n,))
    alpha[ytrain>0] = C * (1-p)
    eta = np.ones(n)
    xi = np.ones(n)
    lambda_svm = 1

    # Initialize barrier parameter mu and dual variable lambda_svm
    mu = (eta.T@alpha+xi.T@(C-alpha))/(2*n)
    mu_vec = [mu]
    # lambda_svm = 0

    # Specify function for matrix-vector product YKY
    MAT = lambda x: kernel_matvec(x, KER, ytrain)

    # Initialize MATcurrent
    # Compute MAT(alpha) for current alpha
    MATcurrent = MAT(alpha)

    # Compute initial infeasibilities, norms & primal/dual tolerances required
    xid = 1 - MATcurrent + lambda_svm * ytrain + mu*1/alpha-mu*1/(C-alpha) # Right hand side first component
    xip = alpha.conj().T @ ytrain # Right hand side second component

    print((eta.T@alpha+eta.T@(C-alpha))/(2*n))
    nrmxid = np.linalg.norm(xid)
    dtol = tol * (1 + np.sqrt(n)) # 1+||c|| in 25 years paper
    nrmxip = np.linalg.norm(xip)
    ptol = tol * (1+0) # 1+||b|| in 25 years paper

    # Set iteration counter
    i = 0
    # Iteration loop for interior point method
    while (i < iter_ip) & (mu > tol or nrmxip > ptol or nrmxid > dtol):

        i = i+1
        # Solve Newton system using MINRES to obtain search direction
        rhs = np.append(xid, xip)
        rhs = np.reshape(rhs,(len(rhs),))

        # initialize counter for GMRES iterations taken
        Giter = 0
        # function to count number of iterations needed in gmres
        def callback_gmres(rk):
            nonlocal Giter
            Giter += 1

        # Prepare the diagonal matrix Theta in equation (5)
        DD = 1/alpha*eta+1/(C-alpha)*xi

        # =============================================================================
        # A as matrix -> see left side of (5) in paper
        Ax = lambda x: hessian_ipm_pd(x,KER,DD,ytrain,n,C,alpha,mu)
        A = LinearOperator((n+1,n+1), Ax)
        # =============================================================================

        # setup SMW preconditioner
        dinv = scipy.sparse.spdiags(1.0/DD, 0, len(DD), len(DD))
        bb = Ldec.conj().T @ (dinv@Ldec)
        aa = np.eye(Ldec.shape[1])
        test = aa+bb
        lu, piv = lu_factor(test)
        PCx= lambda x : SMW_prec(x, n, Ldec, ytrain, DD, test, lu, piv)
        M = LinearOperator((n+1,n+1), PCx)

        [dx,flag] = scipy.sparse.linalg.gmres(A=A, b=rhs, tol=Gtol, restart=Gmaxiter, maxiter=1, M=M, callback=callback_gmres, callback_type='pr_norm')

        # break if the GMRES iterations are too large
        # Store number of Gmres iterations taken
        GMRESiter.append(Giter)
        print("GMRESiters:", GMRESiter, "IPMiters:", i)
        if Giter>99:
            break;

        # Compute new iterates
        dalpha = dx[:-1]
        dlambda = dx[-1]
        deta = mu/alpha-eta-eta/alpha*dalpha
        dxi = mu/(C-alpha)-xi+xi/(C-alpha)*dalpha
        
        ##########################################
        # START LINE SEARCH
        
        # Compute largest value of step such that constraint on alpha, xi, eta hold
        stepL = -1/np.minimum(np.min(dalpha/(alpha+1e-15)),-1)
        stepU = -1/np.minimum(np.min(-dalpha/(C-alpha+1e-15)),-1)
        stepEta = -1/np.minimum(np.min(deta/(eta+1e-15)),-1)
        stepXi = -1/np.minimum(np.min(dxi/(xi+1e-15)),-1)
        
        stepP = np.minimum(stepL,stepU)
        stepD = np.minimum(stepEta,stepXi)
        smax = np.minimum(stepP,stepD)
        
        # Initialize step size
        step = gap * smax
        smin = 0.1 * smax

        # Initialize backtracking parameters a_BT and b_BT
        a_BT = 0.01
        b_BT = 0.8

        # Initialize number of backtracking iterations
        nIterBT = 0
        # Set number of maximum backtracking iterations
        max_iters_BT = 5
        # Initialize Boolean for backtracking routine
        stopCondBackTrack = False
        
        # Compute MAT(dalpha) for update dalpha
        MATupdate = MAT(dalpha)
        
        # Compute evaluation of F in Newton system at current iterates
        F_current = evaluate_F_Newton(alpha, lambda_svm, MATcurrent, ytrain, C, mu)

        # Line Search Routine
        while (stopCondBackTrack == False and nIterBT < max_iters_BT):
            print("initial step LS:", step)
            # Compute updated F for new step size
            F_updated = evaluate_F_Newton(alpha+step*dalpha, lambda_svm+step*dlambda, MATcurrent+step*MATupdate, ytrain, C, mu)
            print("Norm F_current:", np.linalg.norm(F_current))
            print("Norm F_updated:", np.linalg.norm(F_updated))
            # Check if constraints are satisfied
            if (np.linalg.norm(F_updated,ord=np.inf) <= ((1-a_BT*step) * np.linalg.norm(F_current,ord=np.inf))):
                stopCondBackTrack = True
            else:
                step = b_BT * step
                if step < smin:
                    step = smin
                    break
                nIterBT += 1

        print("nIterBT:", nIterBT)
        print("step:", step)
        
        # END LINE SEARCH
        ##########################################
        
        # Update iterates
        alpha = alpha + step*dalpha
        lambda_svm = lambda_svm + step*dlambda
        xi = xi + step*dxi
        eta = eta + step*deta

        # Update MATcurrent
        MATcurrent = MATcurrent + step*MATupdate

        # Update barrier parameter mu
        mu = sigma_br * mu

        # Set lower bound for mu to avoid getting too close to zero barrier
        if (mu < tol/1):
            mu = tol/1
        mu_vec.append(mu)

        # Compute infeasibilities and norms
        xid = 1 - MATcurrent + lambda_svm * ytrain + mu*1/alpha-mu*1/(C-alpha) # Right hand side first component
        xip = alpha.conj().T @ ytrain # Right hand side second component
        nrmxip = np.linalg.norm(xip)
        nrmxid = np.linalg.norm(xid)
        print("norm of stopping mu:",mu)
        print("Norm of the xip:", nrmxip, "norm of xid:", nrmxid)

        if (nrmxid < dtol):
            print("\nIPM has converged!\n")
            break

    return [alpha, GMRESiter, i]
