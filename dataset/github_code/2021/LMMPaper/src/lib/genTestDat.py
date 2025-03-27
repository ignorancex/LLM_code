import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import os
import shutil

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
sys.path.insert(1, os.path.join(sys.argv[0],'..','..','..','lib'))
from npMatrix2d import vech2mat2D

# =============================================================================
# This file contains functions for generating data and calculating the product
# matrices, used by the LMM simulations.
#
# Author: Tom Maullin
# Last edited: 06/04/2020
#
# =============================================================================

# =============================================================================
#
# The below function generates a random testcase according to the linear mixed
# model:
#
#   Y = X\beta + Zb + \epsilon
#
# Where b~N(0,D) and \epsilon ~ N(0,\sigma^2 I)
#
# -----------------------------------------------------------------------------
#
# It takes the following inputs:
#
# -----------------------------------------------------------------------------
#
#   - n (optional): Number of observations. If not provided, a random n will be
#                   selected between 800 and 1200.
#   - p (optional): Number of fixed effects parameters. If not provided, a
#                   random p will be selected between 2 and 10 (an intercept is
#                   automatically included).
#   - nlevels (optional): A vector containing the number of levels for each
#                         random factor, e.g. `nlevels=[3,4]` would mean the
#                         first factor has 3 levels and the second factor has
#                         4 levels. If not provided, default values will be
#                         between 8 and 40.
#   - nraneffs (optional): A vector containing the number of random effects for
#                          each factor, e.g. `nraneffs=[2,1]` would mean the 
#                          first factor has random effects and the second
#                          factor has 1 random effect.
#   - save (optional): Boolean value. If true, the design will be saved.
#   - desInd (optional): Integer value. Index representing which design is
#                        being run. 0 provides a back door option for random
#                        settings.
#   - simInd (optional): Integer value. Index representing which simulation is
#                        being run, only needed for saving.
#   - OutDir (optional): Output directory to save the design to, if saving.
#
# -----------------------------------------------------------------------------
#
# And gives the following outputs:
#
# -----------------------------------------------------------------------------
#
#   - X: A fixed effects design matrix of dimensions (n x p) including a random
#        intercept column (the first column).
#   - Y: A response vector of dimension (n x 1).
#   - Z: A random effects design matrix of size (n x q) where q is equal to the
#        product of nlevels and nraneffs.
#   - nlevels: A vector containing the number of levels for each random factor,
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
#   - beta: The true values of beta used to simulate the response vector.
#   - sigma2: The true value of sigma2 used to simulate the response vector.
#   - D: The random covariance matrix used to simulate b and the response vector.
#   - b: The random effects vector used to simulate the response vector.
#
# -----------------------------------------------------------------------------
def genTestData2D(n=None, p=None, nlevels=None, nraneffs=None, save=False, simInd=None, desInd=0, OutDir=None, factorVectors=None, X=None, Z=None):

    # Check if we have n
    if n is None:

        # If not generate a random n
        n = np.random.randint(800,1200)
    
    # Check if we have p
    if p is None:

        # If not generate a random p
        p = np.random.randint(2,10)

    # Work out number of random factors.
    if nlevels is None and nraneffs is None:

        # If we have neither nlevels or nraneffs, decide on a number of
        # random factors, r.
        r = np.random.randint(2,4)

    # If we have nraneffs use it to work out r
    elif nlevels is None:

        # Work out number of random factors, r
        r = np.shape(nraneffs)[0]

    # If we have nlevels use it to work out r
    else:

        # Work out number of random factors, r
        r = np.shape(nlevels)[0]

    # Check if we need to generate nlevels.
    if nlevels is None:
        
        # Generate random number of levels.
        nlevels = np.random.randint(8,40,r)

    # Check if we need to generate nraneffs.
    if nraneffs is None:
        
        # Generate random number of levels.
        nraneffs = np.random.randint(2,5,r)

    # Assign outdir and simInd if needed:
    if save:
        if not OutDir:
            OutDir = os.getcwd()
        if not simInd:
            simInd = ''
        if not desInd:
            desInd = ''

    # If a design matrix has not been specified make one.
    if X is None:

        # Generate random X.
        X = np.random.randn(n,p)
        
        # Make the first column an intercept
        X[:,0]=1

    # Dictionary of factor vectors
    if factorVectors is None:
        factorVectors = dict()

    # Check if we need to make Z
    if Z is None:

        # Create Z
        # We need to create a block of Z for each level of each factor
        for i in np.arange(r):

            # Covariates for block
            Zdata_factor = np.random.randn(n,nraneffs[i])

            # First factor (WLOG we order this factor so its part of Z is block diagonal
            # - this helps bug testing)
            if i==0:

                # Check if we have a prerecorded factor vector
                if 'i' not in factorVectors:

                    #The first factor should be block diagonal, so the factor indices are grouped
                    factorVec = np.repeat(np.arange(nlevels[i]), repeats=np.floor(n/max(nlevels[i],1)))

                    # Check we have enough indices
                    if len(factorVec) < n:

                        # Quick fix incase rounding leaves empty columns
                        factorVecTmp = np.zeros(n)
                        factorVecTmp[0:len(factorVec)] = factorVec
                        factorVecTmp[len(factorVec):n] = nlevels[i]-1
                        factorVec = np.int64(factorVecTmp)

                        # Crop the factor vector - otherwise have a few too many
                        factorVec = factorVec[0:n]

                    # Give the data an intercept
                    Zdata_factor[:,0]=1

                # Use factor if we have it already
                else:

                    # If we specified a factor vec to use, use it
                    factorVec = factorVectors[i]

                    # Give the data an intercept
                    Zdata_factor[:,0]=1

            else:

                # Check if we have a prerecorded factor vector
                if 'i' not in factorVectors: 
        
                    # The factor is randomly arranged 
                    factorVec = np.random.randint(0,nlevels[i],size=n) 
                    while len(np.unique(factorVec))<nlevels[i]:
                        factorVec = np.random.randint(0,nlevels[i],size=n)

                # Use factor if we have it already
                else:

                    # The factor is randomly arranged 
                    factorVec = factorVectors[i]

            # Save the factor vectors if we don't have it already
            if 'i' not in factorVectors:

                factorVectors[i]=factorVec

            # If outputting, save the factor vector
            if save:
                np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'), factorVec)
                np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'), Zdata_factor)


            # Build a matrix showing where the elements of Z should be
            indicatorMatrix_factor = np.zeros((n,nlevels[i]))
            indicatorMatrix_factor[np.arange(n),factorVec] = 1

            # Need to repeat for this matrix for each random effect/covariate the factor groups 
            indicatorMatrix_factor = np.repeat(indicatorMatrix_factor, nraneffs[i], axis=1)

            # Enter the Z values
            indicatorMatrix_factor[indicatorMatrix_factor==1]=Zdata_factor.reshape(Zdata_factor.shape[0]*Zdata_factor.shape[1])

            # Make sparse
            Zfactor = scipy.sparse.csr_matrix(indicatorMatrix_factor)

            # Put all the factors together
            if i == 0:
                Z = Zfactor
            else:
                Z = scipy.sparse.hstack((Z, Zfactor))

        # Convert Z to dense
        Z = Z.toarray()

    else:

        # If outputting, save the factor vector
        if save:
            for i in range(len(nraneffs)):
                shutil.copyfile(os.path.join(OutDir, 'Sim1_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'),os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zfactor' + str(i) + '.csv'))
                shutil.copyfile(os.path.join(OutDir, 'Sim1_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'),os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Zdata' + str(i) + '.csv'))

    # Make random beta
    if desInd==0:
        beta = np.random.randint(-5,5,p).reshape(p,1)
    else:
        beta = p-np.arange(1,p+1).reshape(p,1)

    # Set sigma to 1 unless generating at random
    if desInd==0:
        # Make random sigma2
        sigma2 = 0.5*np.random.randn()**2
    else:
        sigma2 = 1

    # Make epsilon.
    epsilon = sigma2*np.random.randn(n).reshape(n,1)

    # Make random D
    Ddict = dict()
    Dhalfdict = dict()
    if desInd==0:
    
        # Loop through each factor
        for k in np.arange(r):
      
            # Create a random matrix
            randMat = np.random.uniform(-1,1,(nraneffs[k],nraneffs[k]))

            # Record it as D^{1/2}
            Dhalfdict[k] = randMat

            # Work out D = D^{1/2} @ D^{1/2}'
            Ddict[k] = randMat @ randMat.transpose()

    # Design 1 covariance matrix
    elif desInd==1:

        Ddict[0]=np.eye(2)

    # Design 2 covariance matrix
    elif desInd==2:

        Ddict[0]= vech2mat2D(np.array([[1,0.7,0.5,0.9,0.6,0.8]]).transpose())
        Ddict[1]= np.eye(2)

    # Design 3 covariance matrix
    elif desInd==3:

        Ddict[0]=vech2mat2D(np.array([[1,0.75,0.5,0.25,1,0.75,0.5,1,0.75,1]]).transpose())
        Ddict[1]=vech2mat2D(np.array([[1,0.7,0.5,0.9,0.6,0.8]]).transpose())
        Ddict[2]=np.eye(2)

    # Work out the cholesky factors of Dk (used to simulate b)
    if desInd!=0:
    
        for k in np.arange(r):

            # Record D^{1/2}
            Dhalfdict[k] = np.linalg.cholesky(Ddict[k])


    # Matrix version
    # Construct D and Dhalf in full
    for i in np.arange(r):
      
        # Loop through levels adding blocks to D and the cholesky of D
        for j in np.arange(nlevels[i]):
        
            # If its the first block we're adding, initialize D and Dhalf
            if i == 0 and j == 0:
                D = Ddict[i]
                Dhalf = Dhalfdict[i]

            # Else add to D and Dhalf
            else:
                D = scipy.linalg.block_diag(D, Ddict[i])
                Dhalf = scipy.linalg.block_diag(Dhalf, Dhalfdict[i])


    # Make random b
    q = np.sum(nlevels*nraneffs)
    b = np.random.randn(q).reshape(q,1)

    # Give b the correct covariance structure
    b = Dhalf @ b

    # Generate the response vector
    Y = X @ beta + Z @ b + epsilon

    # If saving output X and Y
    if save:
        np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_X' + '.csv'), X)
        np.savetxt(os.path.join(OutDir, 'Sim' + str(simInd) + '_Design' + str(desInd) + '_Y' + '.csv'), Y)

    # Return values
    return(Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D, factorVectors)


# =============================================================================
#
# The below function generates the product matrices from matrices X, Y and Z.
#
# -----------------------------------------------------------------------------
#
# It takes as inputs:
#
# -----------------------------------------------------------------------------
#
#  - `X`: The design matrix of dimension n times p.
#  - `Y`: The response vector of dimension n times 1.
#  - `Z`: The random effects design matrix of dimension n times q.
#
# -----------------------------------------------------------------------------
#
# It returns as outputs:
#
# -----------------------------------------------------------------------------
#
#  - `XtX`: X transposed multiplied by X.
#  - `XtY`: X transposed multiplied by Y.
#  - `XtZ`: X transposed multiplied by Z.
#  - `YtX`: Y transposed multiplied by X.
#  - `YtY`: Y transposed multiplied by Y.
#  - `YtZ`: Y transposed multiplied by Z.
#  - `ZtX`: Z transposed multiplied by X.
#  - `ZtY`: Z transposed multiplied by Y.
#  - `ZtZ`: Z transposed multiplied by Z.
#
# =============================================================================
def prodMats2D(Y,Z,X):

    # Work out the product matrices
    XtX = X.transpose() @ X
    XtY = X.transpose() @ Y
    XtZ = X.transpose() @ Z
    YtX = XtY.transpose()
    YtY = Y.transpose() @ Y
    YtZ = Y.transpose() @ Z
    ZtX = XtZ.transpose()
    ZtY = YtZ.transpose()
    ZtZ = Z.transpose() @ Z

    # Return product matrices
    return(XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ)