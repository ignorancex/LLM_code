import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import scipy.sparse
import os
np.set_printoptions(threshold=np.inf)
from npMatrix2d import *

# ============================================================================
# 
# This below function performs full Simplified Fisher Scoring for the ACE
# Linear Mixed Model. It is based on the update rules:
#
#               beta = (X'V^(-1)X)^(-1)(X'V^(-1)Y)
#
#                    sigma2E = e'V^(-1)e/n
#
#               vec(\tau2A,\tau2C) = \theta_f + 
#       lam*I(vec(\tau2A,\tau2C))^+ (dl/dvec(\tau2A,\tau2C)
#
# Where:
#  - lam is a scalar stepsize.
#  - \sigma2E is the environmental variance in the ACE model
#  - \tau2A, \tau2C are the A and C variance in the ACE model divided by
#    the E variance in the ACE model
#  - I(vec(\tau2A,\tau2C)) is the Fisher Information matrix of
#    vec(\tau2A,\tau2C).
#  - dl/dvec(\tau2A,\tau2C) is the derivative of the log likelihood with
#    respect to vec(\tau2A,\tau2C). 
#  - e is the residual vector (e=Y-X\beta)
#  - V is the matrix (I+ZDZ')
#
# The name "Simplified" here comes from a convention adopted in (Demidenko 
# 2014).
#
# ----------------------------------------------------------------------------
#
# This function takes as input;
#
# ----------------------------------------------------------------------------
#
# - X: The fixed effects design matrix.
# - Y: The response vector.
# - nlevels: A vector containing the number of levels for each factor,
#            e.g. `nlevels=[3,4]` would mean the first factor has 3
#            levels and the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - tol: Convergence tolerance for the parameter estimation method.
# - KinshipA: A dictionary of kinship matrices for the addtive genetic effect,
#             one corresponding to each family structure type in the model.
# - KinshipC: A dictionary of kinship matrices for the common environmental
#             effect, one corresponding to each family structure type in the
#             model.
# - Constrmat1stDict: A dictionary of constraint matrices. The entry with key 
#                     `k` in the dictionary must map vec(D_k) to \tilde{Tau2}_k.
#                     See Appendix 6.7.3 of the LMM Fisher Scoring paper for 
#                     more information. 
# - Constrmat2nd: A constraint matrix which maps the vector [\tilde{Tau2}_0,...
#                 \tilde{Tau2}_r] to \tau2. See Appendix 6.7.3 of the LMM
#                 Fisher Scoring paper for more information.
# - reml: Restricted maximum likelihood estimation. Default: False.
#
# ----------------------------------------------------------------------------
#
# And returns:
#
# ----------------------------------------------------------------------------
#
#  - `paramVector`: The parameter vector (beta,sigma2,tau2A,tau2C)
#  - `llh`: The log-likelihood value following optimization.
#
# ============================================================================
def fFS_ACE2D(X, Y, nlevels, nraneffs, tol, n, KinshipA, KinshipC, Constrmat1stDict, Constrmat2nd, reml=False):
    
    # ------------------------------------------------------------------------------
    # Product matrices of use
    # ------------------------------------------------------------------------------
    XtX = X.transpose() @ X
    XtY = X.transpose() @ Y
    YtX = Y.transpose() @ X
    YtY = Y.transpose() @ Y

    # ------------------------------------------------------------------------------
    # Useful scalars
    # ------------------------------------------------------------------------------

    # Number of factors, r
    r = len(nlevels)

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Number of fixed effects, p
    p = XtX.shape[0]

    # ------------------------------------------------------------------------------
    # Index variables
    # ------------------------------------------------------------------------------

    # Indices for submatrics corresponding to Dks
    FishIndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)/2) + p + 1)
    FishIndsDk = np.insert(FishIndsDk,0,p+1)

    # Work out D indices (there is one block of D per level)
    Dinds = np.zeros(np.sum(nlevels)+1)
    counter = 0
    for k in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[k]):
            Dinds[counter] = np.concatenate((np.array([0]), np.cumsum(nlevels*nraneffs)))[k] + nraneffs[k]*j
            counter = counter + 1

    # Last index will be missing so add it
    Dinds[len(Dinds)-1]=Dinds[len(Dinds)-2]+nraneffs[-1]
    
    # Make sure indices are ints
    Dinds = np.int64(Dinds)

    # ------------------------------------------------------------------------------
    # Initial estimates
    # ------------------------------------------------------------------------------
    # If we have initial estimates use them.

    # Inital beta
    beta = initBeta2D(XtX, XtY)

    # Work out e'e
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Initial sigma2
    sigma2 = initSigma22D(ete, n)
    sigma2 = np.maximum(sigma2,1e-20) # Prevent hitting boundary
    
    # Initial zero matrix to hold the matrices Skcov(dl/Dk)Sk'
    FDk = np.zeros((2*r,2*r))

    # Initial zero vector to hold the vectors Sk*dl/dDk
    SkdldDk = np.zeros((2*r,1))

    # Initial residuals
    e = Y - X @ beta

    for k in np.arange(r):

        # Get FDk
        FDk[2*k:(2*k+2),2*k:(2*k+2)]= nlevels[k]*Constrmat1stDict[k] @ Constrmat1stDict[k].transpose()

        # Get the indices for the factors 
        Ik = fac_indices2D(k, nlevels, nraneffs)

        # Get Ek
        Ek = e[Ik,:].reshape((nlevels[k],nraneffs[k])).transpose()

        # Get Sk*dl/dDk
        SkdldDk[2*k:(2*k+2),:] = Constrmat1stDict[k] @ mat2vec2D(nlevels[k]-Ek @ Ek.transpose()/sigma2)

    # Initial vec(sigma^2A/sigma^2E, sigma^2D/sigma^2E)
    dDdAD = 2*Constrmat2nd
    tau2 = np.linalg.pinv(dDdAD @ FDk @ dDdAD.transpose()) @ dDdAD @ SkdldDk

    # Inital D (dict version)
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = tau2[0,0]**2*KinshipA[k] + tau2[1,0]**2*KinshipC[k]

    # ------------------------------------------------------------------------------
    # Obtain (I+D)^{-1}
    # ------------------------------------------------------------------------------
    invIplusDdict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        invIplusDdict[k] = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])

    # ------------------------------------------------------------------------------
    # Initial lambda and likelihoods
    # ------------------------------------------------------------------------------
    # Step size lambda
    lam = 1
    
    # Initial log likelihoods
    llhprev = np.inf
    llhcurr = -np.inf

    # ------------------------------------------------------------------------------
    # Precalculated Kronecker sums
    # ------------------------------------------------------------------------------

    # Initialize empty dictionaries
    XkXdict = dict()
    XkYdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        # Sum XkY
        XkYdict[k] = np.zeros((p,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

            # Add to running sum
            XkYdict[k] = XkYdict[k] + np.kron(Y[Ikj,:].transpose(),X[Ikj,:].transpose())

    # ------------------------------------------------------------------------------
    # Iteration.
    # ------------------------------------------------------------------------------
    # Number of iterations
    nit = 0
    while np.abs(llhprev-llhcurr)>tol:

        # Change current likelihood to previous
        llhprev = llhcurr

        # Number of iterations
        nit = nit+1

        #---------------------------------------------------------------------------
        # Update beta
        #---------------------------------------------------------------------------
        # Work out X'V^(-1)X and X'V^(-1)Y
        XtinvVX = np.zeros((p,p))
        XtinvVY = np.zeros((p,1))


        # Loop through levels and factors
        for k in np.arange(r):

            XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(invIplusDdict[k]),shape=np.array([p,p]))
            XtinvVY = XtinvVY + vec2mat2D(XkYdict[k] @ mat2vec2D(invIplusDdict[k]),shape=np.array([p,1]))

        beta = np.linalg.solve(forceSym2D(XtinvVX), XtinvVY)

        #---------------------------------------------------------------------------
        # Update Residuals, e
        #---------------------------------------------------------------------------
        e = Y - X @ beta
        ete = e.transpose() @ e

        #---------------------------------------------------------------------------
        # Update sigma^2
        #---------------------------------------------------------------------------

        # Initial e'V^{-1}e
        etinvVe = np.zeros((1,1))

        # Loop through levels and factors
        for k in np.arange(r):
            for j in np.arange(nlevels[k]):

                # Get the indices for the factors 
                Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

                # Add to sums
                etinvVe = etinvVe + e[Ikj,:].transpose() @ invIplusDdict[k] @ e[Ikj,:]

        # Calculate sigma^2
        if not reml:
            sigma2 = 1/n*etinvVe
        else:
            sigma2 = 1/(n-p)*etinvVe
        sigma2 = np.maximum(sigma2,1e-20) # Prevent hitting boundary

        # Initial zero matrix to hold F
        F = np.zeros((2,2))

        # Initial zero vector to hold the vectors Sk*dl/dDk
        S = np.zeros((2,1))

        for k in np.arange(len(nraneffs)):

            #-----------------------------------------------------------------------
            # Work out derivative of D_k
            #-----------------------------------------------------------------------

            # Get the indices for the factors
            Ik = fac_indices2D(k, nlevels, nraneffs)

            # Get Ek
            Ek = e[Ik,:].reshape((nlevels[k],nraneffs[k])).transpose()
            
            # # Calculate S'dl/dDk
            # SkdldDk2[2*k:(2*k+2),:] =  Constrmat1stDict[k] @ mat2vec2D((invIplusDdict[k] @ Ek2 @ Ek2.transpose() @ invIplusDdict[k]/sigma22[0,0])-nlevels[k]*invIplusDdict[k])

            if not reml:
                S = S + Constrmat1stDict[k] @ mat2vec2D(forceSym2D((invIplusDdict[k] @ Ek @ Ek.transpose() @ invIplusDdict[k]/sigma2[0,0])-nlevels[k]*invIplusDdict[k]))
            else:
                CurrentS = mat2vec2D(forceSym2D((invIplusDdict[k] @ Ek @ Ek.transpose() @ invIplusDdict[k]/sigma2[0,0])-nlevels[k]*invIplusDdict[k]))
                CurrentS =  CurrentS + np.kron(invIplusDdict[k],invIplusDdict[k]) @ XkXdict[k].transpose() @ mat2vec2D(np.linalg.pinv(XtinvVX))
                S = S + Constrmat1stDict[k] @ CurrentS

            #-----------------------------------------------------------------------
            # Work out covariance of derivative of D_k
            #-----------------------------------------------------------------------

            # Work out (I+Dk)^(-1) \otimes (I+Dk)^(-1)
            kronTerm = np.kron(invIplusDdict[k],invIplusDdict[k])

            # Get F for this term
            F = F + forceSym2D(nlevels[k]*Constrmat1stDict[k] @ kronTerm @ Constrmat1stDict[k].transpose())

        #-----------------------------------------------------------------------
        # Update step
        #-----------------------------------------------------------------------

        tau2 = tau2 + forceSym2D(0.5*lam*np.linalg.pinv(forceSym2D(F)*(tau2 @ tau2.transpose()))) @ (tau2*S)
        
        #-----------------------------------------------------------------------
        # Add D_k back into D
        #-----------------------------------------------------------------------
        # Inital D (dict version)
        Ddict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            Ddict[k] = forceSym2D(tau2[0,0]**2*KinshipA[k] + tau2[1,0]**2*KinshipC[k])

        # ------------------------------------------------------------------------------
        # Obtain (I+D)^{-1}
        # ------------------------------------------------------------------------------
        invIplusDdict = dict()
        for k in np.arange(len(nraneffs)):
            # Construct D using sigma^2A and sigma^2D
            invIplusDdict[k] = forceSym2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]))

        # --------------------------------------------------------------------------
        # Precautionary
        # --------------------------------------------------------------------------
        # Check sigma2 hasn't hit a boundary
        if sigma2<0:
            sigma2=1e-10

        # --------------------------------------------------------------------------
        # Update step size and likelihood
        # --------------------------------------------------------------------------

        # Update e'V^(-1)e
        etinvVe = np.zeros((1,1))
        # Loop through levels and factors
        for k in np.arange(r):
            for j in np.arange(nlevels[k]):

                # Get the indices for the factors 
                Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

                # Add to sums
                etinvVe = etinvVe + e[Ikj,:].transpose() @ invIplusDdict[k] @ e[Ikj,:]

        # Work out log|V| using the fact V is block diagonal
        logdetV = 0
        for k in np.arange(r):
            logdetV = logdetV - nlevels[k]*np.prod(np.linalg.slogdet(invIplusDdict[k]))

        # Work out the log likelihood
        llhcurr = -0.5*(n*np.log(sigma2)+(1/sigma2)*etinvVe + logdetV)

        if reml:
            logdet = np.linalg.slogdet(XtinvVX)
            llhcurr = llhcurr - 0.5*logdet[0]*logdet[1] + 0.5*p*np.log(sigma2)

        # Update the step size
        if llhprev>llhcurr:
            lam = lam/2

    # ------------------------------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------------------------------
    paramVector = np.concatenate((beta, np.sqrt(sigma2), tau2))
        
    return(paramVector, llhcurr)

# ============================================================================
#
# The below function estimates the degrees of freedom for an T statistic using
# a Sattherthwaite approximation method. For, a contrast matrix L, this 
# estimate is given by:
#
#    v = 2(Var(L\beta)^2)/(d'I^{-1}d)
#
# Where d is the derivative of Var(L\beta) with respect to the variance 
# parameter vector \theta = (\sigma^2, \tauA^2, \tauC^2) and I is the
# Fisher Information matrix of \theta^{ACE}.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - `L`: A contrast vector.
# - `paramVec`: Final estimates of the parameter vector.
# - `nlevels`: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - KinshipA: A dictionary of kinship matrices for the addtive genetic effect,
#             one corresponding to each family structure type in the model.
# - KinshipC: A dictionary of kinship matrices for the common environmental
#             effect, one corresponding to each family structure type in the
#             model.
# - Constrmat1stDict: A dictionary of constraint matrices. The entry with key 
#                     `k` in the dictionary must map vec(D_k) to \tilde{Tau2}_k.
#                     See Appendix 6.7.3 of the LMM Fisher Scoring paper for 
#                     more information. 
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `df`: The Sattherthwaithe degrees of freedom estimate.
#
# ============================================================================
def get_swdf_ACE_T2D(L, paramVec, X, nlevels, nraneffs, KinshipA, KinshipC, Constrmat1stDict): 

    # Work out n and p
    n = X.shape[0]
    p = X.shape[1]

    # Work out beta, sigma2 and the vector of variance components
    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,0]**2
    tau2 = paramVec[(p+1):,:]

    # Get D in dictionary form
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = tau2[0,0]**2*KinshipA[k] + tau2[1,0]**2*KinshipC[k]

    # Get S^2 (= Var(L\beta))
    S2 = get_varLB_ACE_2D(L, X, Ddict, sigma2, nlevels, nraneffs)

    # Get derivative of S^2
    dS2 = get_dS2_ACE_2D(L, X, Ddict, tau2, sigma2, nlevels, nraneffs, Constrmat1stDict)

    # Get Fisher information matrix
    InfoMat = get_InfoMat_ACE_2D(Ddict, tau2, sigma2, n, nlevels, nraneffs, Constrmat1stDict)

    # Calculate df estimator
    df = 2*(S2**2)/(dS2.transpose() @ np.linalg.solve(InfoMat, dS2))

    # Return df
    return(df)

# ============================================================================
#
# The below function calculates the (in most applications, scalar) variance
# of L\beta.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - L: A contrast vector (L can also be a matrix, but this isn't often the
#      case in practice when using this function).
# - X: The fixed effects design matrix.
# - Ddict: Dictionary version of the random effects variance-covariance
#          matrix.
# - sigma2: The fixed effects variance (\sigma^2 in the previous notation).
# - nlevels: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - varLB: The (usually scalar) variance of L\beta.
#
# ============================================================================
def get_varLB_ACE_2D(L, X, Ddict, sigma2, nlevels, nraneffs):

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    varLB = L @ get_covB_ACE_2D(X, Ddict, sigma2, nlevels, nraneffs) @ L.transpose()

    # Return result
    return(varLB)

# ============================================================================
#
# The below function gives the covariance matrix of the beta estimates.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - L: A contrast vector (L can also be a matrix, but this isn't often the
#      case in practice when using this function).
# - X: The fixed effects design matrix.
# - Ddict: Dictionary version of the random effects variance-covariance
#          matrix.
# - sigma2: The fixed effects variance (\sigma^2 in the previous notation).
# - nlevels: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - covB: The covariance of the beta estimates.
#
# ============================================================================
def get_covB_ACE_2D(X, Ddict, sigma2, nlevels, nraneffs):

    # Work out p and r
    p = X.shape[1]
    r = len(nlevels)

    # Work out sum over j of X_(k,j) kron X_(k,j), for each k
    XkXdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

    # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
    XtinvVX = np.zeros((p,p))

    # Loop through levels and factors
    for k in np.arange(r):

        XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])),shape=np.array([p,p]))

    # Work out var(LB) = L'(X'V^{-1}X)^{-1}L
    covB = np.linalg.pinv(XtinvVX)

    # Calculate sigma^2(X'V^{-1}X)^(-1)
    covB = sigma2*covB

    # Return result
    return(covB)


# ============================================================================
#
# The below function calculates the derivative of Var(L\beta) with respect to
# the variance parameter vector \theta = (\sigma^2, \tauA^2, \tauC^2).
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - L: A contrast vector (L can also be a matrix, but this isn't often the
#      case in practice when using this function).
# - X: The fixed effects design matrix.
# - Ddict: Dictionary version of the random effects variance-covariance
#          matrix.
# - tau2: The vector of scaled variance estimates; (\tauA^2, \tauC^2)
# - sigma2: The fixed effects variance (\sigma^2 in the previous notation).
# - nlevels: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - Constrmat1stDict: A dictionary of constraint matrices. The entry with key 
#                     `k` in the dictionary must map vec(D_k) to \tilde{Tau2}_k.
#                     See Appendix 6.7.3 of the LMM Fisher Scoring paper for 
#                     more information. 
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - `dS2`: The derivative of var(L\beta) with respect to \theta.
#
# ============================================================================
def get_dS2_ACE_2D(L, X, Ddict, tau2, sigma2, nlevels, nraneffs, Constrmat1stDict):

    # Work out r
    r = len(nlevels)

    # Work out p
    p = X.shape[1]

    # Work out sum over j of X_(k,j) kron X_(k,j), for each k
    XkXdict = dict()

    # Loop through levels and factors
    for k in np.arange(r):

        # Get qk
        qk = nraneffs[k]

        # Sum XkX
        XkXdict[k] = np.zeros((p**2,qk**2))

        for j in np.arange(nlevels[k]):

            # Indices for level j of factor k
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Add to running sum
            XkXdict[k] = XkXdict[k] + np.kron(X[Ikj,:].transpose(),X[Ikj,:].transpose())

    # Work out X'V^(-1)X as matrix reshape of (sum over k of ((sum_j X_(k,j) kron X_(k,j))vec(D_k)))
    XtinvVX = np.zeros((p,p))

    # Loop through levels and factors
    for k in np.arange(r):

        # Calculating X'V^{-1}X
        XtinvVX = XtinvVX + vec2mat2D(XkXdict[k] @ mat2vec2D(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k])),shape=np.array([p,p]))

    # New empty array for differentiating S^2 wrt (sigma2, vech(D1),...vech(Dr)).
    dS2 = np.zeros((3,1))

    # Work of derivative wrt to sigma^2
    dS2dsigma2 = L @ np.linalg.pinv(XtinvVX) @ L.transpose()

    # Add to dS2
    dS2[0:1,0] = dS2dsigma2.reshape(dS2[0:1,0].shape)

    # Now we need to work out ds2dVech(Dk)
    for k in np.arange(len(nraneffs)):

        # Initialize an empty zeros matrix
        dS2dvechDk = np.zeros((nraneffs[k]**2,1))

        for j in np.arange(nlevels[k]):

            # Get the indices for this level and factor.
            Ikj = faclev_indices2D(k, j, nlevels, nraneffs)

            # Work out Z_(k,j)'V^{-1}X
            ZkjtiVX = np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]) @ X[Ikj,:]

            # Work out the term to put into the kronecker product
            # K = Z_(k,j)'V^{-1}X(X'V^{-1})^{-1}L'
            K = ZkjtiVX @ np.linalg.pinv(XtinvVX) @ L.transpose()

            # Sum terms
            dS2dvechDk = dS2dvechDk + mat2vec2D(np.kron(K,K.transpose()))

        # Multiply by sigma^2
        dS2dvechDk = sigma2*dS2dvechDk

        # Add to dS2
        dS2[1:,0:1] = dS2[1:,0:1] + Constrmat1stDict[k] @ dS2dvechDk.reshape((nraneffs[k]**2,1))

    # Multiply by 2tau^2 elementwise
    dS2[1:,0:1] = 2*tau2*dS2[1:,0:1]

    return(dS2)


# ============================================================================
#
# The below function calculates the derivative of Var(L\beta) with respect to
# the variance parameter vector \theta = (\sigma^2, \tau_A^2, \tau_C^2).
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - Ddict: Dictionary version of the random effects variance-covariance
#          matrix.
# - tau2: The vector of scaled variance estimates; (\tauA^2, \tauC^2)
# - sigma2: The fixed effects variance (\sigma^2 in the previous notation).
# - nlevels: A vector containing the number of levels for each factor, e.g.
#              `nlevels=[3,4]` would mean the first factor has 3 levels and
#              the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - Constrmat1stDict: A dictionary of constraint matrices. The entry with key 
#                     `k` in the dictionary must map vec(D_k) to \tilde{Tau}_k.
#                     See Appendix 6.7.3 of the LMM Fisher Scoring paper for 
#                     more information. 
# - n: The total number of observations.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - FisherInfoMat: The Fisher information matrix of \theta.
#
# ============================================================================
def get_InfoMat_ACE_2D(Ddict, tau2, sigma2, n, nlevels, nraneffs, Constrmat1stDict):

    # Number of random effects, q
    q = np.sum(np.dot(nraneffs,nlevels))

    # Index variables
    # ------------------------------------------------------------------------------

    # Initialize FIsher Information matrix
    FisherInfoMat = np.zeros((3,3))

    # Covariance of dl/dsigma2
    C = n/(2*sigma2**2)

    # Add dl/dsigma2 covariance
    FisherInfoMat[0,0] = C

    H = np.zeros((2,1))

    # Get H= cov(dl/sigmaE^2, dl/((sigmaA,sigmaD)/sigmaE))
    for k in np.arange(len(nraneffs)):

        # Get covariance of dldsigma and dldD      
        H = H + Constrmat1stDict[k] @ get_covdldDkdsigma2_ACE_2D(k, sigma2, nlevels, nraneffs, Ddict).reshape((nraneffs[k]**2,1))

    # Assign to the relevant block
    FisherInfoMat[1:,0:1] = 2*tau2*H
    FisherInfoMat[0:1,1:] = FisherInfoMat[1:,0:1].transpose()

    # Initial zero matrix to hold F
    F = np.zeros((2,2))

    for k in np.arange(len(nraneffs)):

        #-----------------------------------------------------------------------
        # Work out covariance of derivative of D_k
        #-----------------------------------------------------------------------

        # Work out (I+Dk)^(-1) \otimes (I+Dk)^(-1)
        kronTerm = np.kron(np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]),np.linalg.pinv(np.eye(nraneffs[k])+Ddict[k]))

        # Get F for this term
        F = F + forceSym2D(nlevels[k]*Constrmat1stDict[k] @ kronTerm @ Constrmat1stDict[k].transpose())

    # Multiply by 2tau2 elementwise on both sides
    F = 2*forceSym2D(F)*(tau2 @ tau2.transpose())

    # Assign to the relevant block
    FisherInfoMat[1:, 1:] = F

    # Return result
    return(FisherInfoMat)


# ============================================================================
#
# The below function calculates the covariance between the derivative of the 
# log likelihood with respect to vech(D_k) and the derivative with respect to 
# \sigma^2.
#
# ----------------------------------------------------------------------------
#
# This function takes the following inputs:
#
# ----------------------------------------------------------------------------
#
# - k: The number of the first factor (k in the above notation).
# - sigma2: The fixed effects variance (\sigma^2 in the above notation).
# - nlevels: A vector containing the number of levels for each factor, e.g.
#            `nlevels=[3,4]` would mean the first factor has 3 levels and 
#            the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - Ddict: Dictionary version of the random effects variance-covariance
#          matrix.
#
# ----------------------------------------------------------------------------
#
# It returns as outputs:
#
# ----------------------------------------------------------------------------
#
# - covdldDdldsigma2: The covariance between the derivative of the log 
#                     likelihood with respect to vech(D_k) and the 
#                     derivative with respect to \sigma^2.
# - ZtZmat: The sum over j of Z_{(k,j)}'Z_{(k,j)}. This only need be 
#           calculated once so can be stored and re-entered for each
#           iteration.
#
# ============================================================================
def get_covdldDkdsigma2_ACE_2D(k, sigma2, nlevels, nraneffs, Ddict):

    # Get the indices for the factors 
    Ik = fac_indices2D(k, nlevels, nraneffs)

    # Work out lk
    lk = nlevels[k]

    # Work out block size
    qk = nraneffs[k]

    # Obtain sum of Rk = lk*(I+Dk)^(-1)
    RkSum = lk*np.linalg.pinv(np.eye(qk)+Ddict[k])

    # save and return
    covdldDdldsigma2 = 1/(2*sigma2) * mat2vec2D(RkSum)  

    return(covdldDdldsigma2)


# ============================================================================
#
# The below function calculates the approximate T statistic for a null
# hypothesis test, H0:L\beta == 0 vs H1: L\beta != 0. The T statistic is given
# by:
#
#     T = L\beta/s.e.(L\beta)
#
# Where s.e. represents standard error.
#
# ----------------------------------------------------------------------------
#
# This function takes in the following inputs:
#
# ----------------------------------------------------------------------------
#
# - L: A contrast vector.
# - X: The fixed effects design matrix.
# - paramVec: Final estimates of the parameter vector.
# - KinshipA: A dictionary of kinship matrices for the addtive genetic effect,
#             one corresponding to each family structure type in the model.
# - KinshipC: A dictionary of kinship matrices for the common environmental
#             effect, one corresponding to each family structure type in the
#             model.
# - nlevels: A vector containing the number of levels for each factor, e.g.
#            `nlevels=[3,4]` would mean the first factor has 3 levels and
#            the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
#
# ----------------------------------------------------------------------------
#
# And gives the following output:
#
# ----------------------------------------------------------------------------
#
# - T: T statistic.
#
# ============================================================================
def get_T_ACE_2D(L, X, paramVec, KinshipA, KinshipC, nlevels, nraneffs):

    # Work out n and p
    n = X.shape[0]
    p = X.shape[1]

    # Work out beta, sigma2 and the vector of variance components
    beta = paramVec[0:p,:]
    sigma2 = paramVec[p,0]**2
    tau2 = paramVec[(p+1):,:]

    # Get D in dictionary form
    Ddict = dict()
    for k in np.arange(len(nraneffs)):
        # Construct D using sigma^2A and sigma^2D
        Ddict[k] = tau2[0,0]**2*KinshipA[k] + tau2[1,0]**2*KinshipC[k]
    
    # Work out the rank of L
    rL = np.linalg.matrix_rank(L)

    # Work out Lbeta
    LB = L @ beta

    # Work out se(T)
    varLB = get_varLB_ACE_2D(L, X, Ddict, sigma2, nlevels, nraneffs)

    # Work out T
    T = LB/np.sqrt(varLB)

    # Return T
    return(T)