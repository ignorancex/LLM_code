import os
import sys
import numpy as np
import pandas as pd
import time
import scipy.sparse
import scipy.sparse.linalg
from scipy import stats
from scipy.optimize import minimize

np.set_printoptions(threshold=sys.maxsize)

# Add lib to the python path.
from genTestDat import genTestData2D, prodMats2D
from est2d import *
from est3d import *
from npMatrix2d import *
from npMatrix3d import *

# ==================================================================================
#
# The below code runs multiple simulations in serial. It takes the following inputs:
#
# ----------------------------------------------------------------------------------
#
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - OutDir: The output directory.
# - nsim: Number of simulations (default=1000)
# - mode: String indicating whether to run parameter estimation simulations (mode=
#         'param') or T statistic simulations (mode='Tstat').
# - REML: Boolean indicating whether to use ML or ReML estimation. 
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def sim2D(desInd, OutDir, nsim=1000, mode='param', REML=False):

    # Loop through and run simulations
    for simInd in range(1,nsim+1):
        runSim(simInd, desInd, OutDir, mode, REML)


# ==================================================================================
#
# The below simulates random test data and runs all methods described in the LMM 
# paper on the simulated data. It requires the following inputs:
#
# ----------------------------------------------------------------------------------
#
# - SimInd: An index to represent the simulation. All output for this simulation will
#           be saved in files with the index specified by this argument. The
#           simulation with index 1 will also perform any necessary additional setup
#           and should therefore be run before any others.
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - OutDir: The output directory.
# - mode: String indicating whether to run parameter estimation simulations (mode=
#         'param') or T statistic simulations (mode='Tstat').
# - REML: Boolean indicating whether to use ML or ReML estimation. 
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def runSim(simInd, desInd, OutDir, mode='param', REML=False):

    # Make sure simInd is an int
    simInd = int(simInd)
        
    #===============================================================================
    # Setup
    #===============================================================================

    # Decide whether we wish to run T statistics/degrees of freedom estimation
    if mode=='param':
        runDF = False
    else:
        runDF = True

    # Different designs
    if desInd==1:
        nlevels = np.array([50])
        nraneffs = np.array([2])
    if desInd==2:
        nlevels = np.array([50,25])
        nraneffs = np.array([3,2])
    if desInd==3:
        nlevels = np.array([100,30,10])
        nraneffs = np.array([4,3,2])

    # Number of observations
    n = 1000

    # If we are doing a degrees of freedom simulation, create the factor vectors, X and Z if 
    # this is the first run. These will then be used across all following simulations. If we
    # are doing a simulation to look at parameter estimation, we recreate the design on every
    # run as our focus is to stress test the performance of the algorithms, rather than compare
    # performance of one specific model in particular. 
    if simInd == 1 or not runDF:

        # Delete any factor vectors from a previous batch of simulations.
        if runDF:
            for i in range(len(nlevels)):

                if os.path.isfile(os.path.join(OutDir, 'fv_' + str(desInd) + '_' + str(i) + '.csv')):

                    os.remove(os.path.join(OutDir, 'fv_' + str(desInd) + '_' + str(i) + '.csv'))

        fvs = None
        X = None
        Z = None

    # Otherwise read the factor vectors, X and Z in from file.
    else:

        # Initialize empty factor vectors dict
        fvs = dict()

        # Loop through factors and save factor vectors
        for i in range(len(nlevels)):

            fvs[i] = pd.io.parsers.read_csv(os.path.join(OutDir, 'fv_' + str(desInd) + '_' + str(i) + '.csv'), header=None).values

        X = pd.io.parsers.read_csv(os.path.join(OutDir, 'X_' + str(desInd)  + '.csv'), header=None).values
        Z = pd.io.parsers.read_csv(os.path.join(OutDir, 'Z_' + str(desInd)  + '.csv'), header=None).values

    # Generate test data
    Y,X,Z,nlevels,nraneffs,beta,sigma2,b,D, fvs = genTestData2D(n=n, p=5, nlevels=nlevels, nraneffs=nraneffs, save=True, simInd=simInd, desInd=desInd, OutDir=OutDir, factorVectors=fvs, X=X, Z=Z)

    # Save the new factor vectors if this is the first run.
    if simInd == 1 and runDF:

        # Loop through the factors saving them
        for i in range(len(nlevels)):

            pd.DataFrame(fvs[i]).to_csv(os.path.join(OutDir, 'fv_' + str(desInd) + '_' + str(i) + '.csv'), index=False, header=None)

        pd.DataFrame(X).to_csv(os.path.join(OutDir, 'X_' + str(desInd) + '.csv'), index=False, header=None)
        pd.DataFrame(Z).to_csv(os.path.join(OutDir, 'Z_' + str(desInd) + '.csv'), index=False, header=None)

    # Work out number of observations, parameters, random effects, etc
    n = X.shape[0]
    p = X.shape[1]
    q = np.sum(nraneffs*nlevels)
    qu = np.sum(nraneffs*(nraneffs+1)//2)
    r = nlevels.shape[0]

    # Tolerance
    tol = 1e-6

    # Work out factor indices.
    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to dict
    Ddict=dict()
    for k in np.arange(len(nlevels)):

        Ddict[k] = D[facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])]

    # Get the product matrices
    XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ = prodMats2D(Y,Z,X)

    # -----------------------------------------------------------------------------
    # Create empty data frame for results:
    # -----------------------------------------------------------------------------

    # Row indices
    indexVec = np.array(['Time', 'nit', 'llh'])
    for i in np.arange(p):

        indexVec = np.append(indexVec, 'beta'+str(i+1))

    # Sigma2
    indexVec = np.append(indexVec, 'sigma2')

    # Dk
    for k in np.arange(r):
        for j in np.arange(nraneffs[k]*(nraneffs[k]+1)//2):
            indexVec = np.append(indexVec, 'D'+str(k+1)+','+str(j+1))

    # Sigma2*Dk
    for k in np.arange(r):
        for j in np.arange(nraneffs[k]*(nraneffs[k]+1)//2):
            indexVec = np.append(indexVec, 'sigma2*D'+str(k+1)+','+str(j+1))

    # If we're doing a T statistic simulation add the T statistics, p values and 
    # degrees of freedom rows to the dataframe.
    if runDF:
        # T value p value and Satterthwaite degrees of freedom estimate.
        indexVec = np.append(indexVec,'T')
        indexVec = np.append(indexVec,'p')
        indexVec = np.append(indexVec,'swdf')

    # Construct dataframe
    results = pd.DataFrame(index=indexVec, columns=['Truth', 'FS', 'fFS', 'SFS', 'fSFS', 'cSFS'])

    # ------------------------------------------------------------------------------------
    # Truth
    # ------------------------------------------------------------------------------------

    # Default time and number of iterations
    results.at['Time','Truth']=0
    results.at['nit','Truth']=0

    # Construct parameter vector
    paramVec_true = beta[:]
    paramVec_true = np.concatenate((paramVec_true,np.array(sigma2).reshape(1,1)),axis=0)

    # Add D to parameter vector
    facInds = np.cumsum(nraneffs*nlevels)
    facInds = np.insert(facInds,0,0)

    # Convert D to vector
    for k in np.arange(len(nlevels)):

        vechD = mat2vech2D(D[facInds[k]:(facInds[k]+nraneffs[k]),facInds[k]:(facInds[k]+nraneffs[k])])/sigma2
        paramVec_true = np.concatenate((paramVec_true,vechD),axis=0)

    # Add results to parameter vector
    for i in np.arange(3,p+qu+4):

        results.at[indexVec[i],'Truth']=paramVec_true[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'Truth']=paramVec_true[p,0]*paramVec_true[i-3,0]

    # Matrices needed for
    Zte = ZtY - ZtX @ beta
    ete = ssr2D(YtX, YtY, XtX, beta)
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)

    # True log likelihood
    llh = llh2D(n, ZtZ, Zte, ete, sigma2, DinvIplusZtZD,D,REML,XtX,XtZ,ZtX)[0,0]

    # Add back on constant term
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Add ground truth log likelihood
    results.at['llh','Truth']=llh

    # Get the ground truth degrees of freedom if running a T statistic simulation
    if runDF:

        # Contrast vector (1 in last place 0 elsewhere)
        L = np.zeros(p)
        L[-1] = 1
        L = L.reshape(1,p)

        v = groundTruth_TDF(X, Z, beta, sigma2, D, L, nlevels, nraneffs, tol)
        results.at[indexVec[p+6+2*qu],'Truth']=v[0,0]


    #===============================================================================
    # fSFS
    #===============================================================================

    # Get the indices for the individual random factor covariance parameters.
    DkInds = np.zeros(len(nlevels)+1)
    DkInds[0]=np.int(p+1)
    for k in np.arange(len(nlevels)):
        DkInds[k+1] = np.int(DkInds[k] + nraneffs[k]*(nraneffs[k]+1)//2)

    # Run Full Simplified Fisher Scoring
    t1 = time.time()
    paramVector_fSFS,_,nit,llh = fSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, init_paramVector=None)
    t2 = time.time()

    # Add back on constant term for llh
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Record Time and number of iterations
    results.at['Time','fSFS']=t2-t1
    results.at['nit','fSFS']=nit
    results.at['llh','fSFS']=llh

    # Record parameters
    for i in np.arange(3,p+qu+4):

        results.at[indexVec[i],'fSFS']=paramVector_fSFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'fSFS']=paramVector_fSFS[p,0]*paramVector_fSFS[i-3,0]
         
    # If running a T statistic simulation...
    if runDF:

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_fSFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu],'fSFS']=T[0,0]
        results.at[indexVec[p+5+2*qu],'fSFS']=Pval[0,0]
        results.at[indexVec[p+6+2*qu],'fSFS']=df[0,0]

    #===============================================================================
    # cSFS
    #===============================================================================

    # Run Cholesky Simplified Fisher Scoring
    t1 = time.time()
    paramVector_cSFS,_,nit,llh = cSFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, init_paramVector=None)
    t2 = time.time()

    # Add back on constant term for llh
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Record time and number of iterations
    results.at['Time','cSFS']=t2-t1
    results.at['nit','cSFS']=nit
    results.at['llh','cSFS']=llh
    
    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'cSFS']=paramVector_cSFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'cSFS']=paramVector_cSFS[p,0]*paramVector_cSFS[i-3,0]

    # If running a T statistic simulation...
    if runDF:

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_cSFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu],'cSFS']=T[0,0]
        results.at[indexVec[p+5+2*qu],'cSFS']=Pval[0,0]
        results.at[indexVec[p+6+2*qu],'cSFS']=df[0,0]

    #===============================================================================
    # FS
    #===============================================================================

    # Run Fisher Scoring
    t1 = time.time()
    paramVector_FS,_,nit,llh = FS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, init_paramVector=None)
    t2 = time.time()

    # Add back on constant term for llh
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Record time and number of iterations
    results.at['Time','FS']=t2-t1
    results.at['nit','FS']=nit
    results.at['llh','FS']=llh
    
    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'FS']=paramVector_FS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'FS']=paramVector_FS[p,0]*paramVector_FS[i-3,0]

    # If running a T statistic simulation...
    if runDF:

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_FS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu],'FS']=T[0,0]
        results.at[indexVec[p+5+2*qu],'FS']=Pval[0,0]
        results.at[indexVec[p+6+2*qu],'FS']=df[0,0]

    #===============================================================================
    # SFS
    #===============================================================================

    # Run Simplified Fisher Scoring
    t1 = time.time()
    paramVector_SFS,_,nit,llh = SFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, init_paramVector=None)
    t2 = time.time()

    # Add back on constant term for llh
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Record time and number of iterations
    results.at['Time','SFS']=t2-t1
    results.at['nit','SFS']=nit
    results.at['llh','SFS']=llh

    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'SFS']=paramVector_SFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'SFS']=paramVector_SFS[p,0]*paramVector_SFS[i-3,0]

    # If running a T statistic simulation...
    if runDF:

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_SFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu],'SFS']=T[0,0]
        results.at[indexVec[p+5+2*qu],'SFS']=Pval[0,0]
        results.at[indexVec[p+6+2*qu],'SFS']=df[0,0]

    #===============================================================================
    # fFS
    #===============================================================================

    # Run Full Fisher Scoring
    t1 = time.time()
    paramVector_fFS,_,nit,llh = fFS2D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol, n, reml=REML, init_paramVector=None)
    t2 = time.time()

    # Add back on constant term for llh
    if REML:
        llh = llh - (n-p)/2*np.log(2*np.pi)
    else:
        llh = llh - n/2*np.log(2*np.pi)

    # Record time and number of iterations
    results.at['Time','fFS']=t2-t1
    results.at['nit','fFS']=nit
    results.at['llh','fFS']=llh

    # Save parameters
    for i in np.arange(3,p+qu+4):
        results.at[indexVec[i],'fFS']=paramVector_fFS[i-3,0]

    # Record D*sigma2
    for i in np.arange(4+p,p+qu+4):
        results.at[indexVec[i+qu],'fFS']=paramVector_fFS[p,0]*paramVector_fFS[i-3,0]

    # If running a T statistic simulation...
    if runDF:

        # Get T statistic, p value and Satterthwaite degrees of freedom
        T,Pval,df = simT(paramVector_fFS, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n)
        results.at[indexVec[p+4+2*qu],'fFS']=T[0,0]
        results.at[indexVec[p+5+2*qu],'fFS']=Pval[0,0]
        results.at[indexVec[p+6+2*qu],'fFS']=df[0,0]

    # Save results
    results.to_csv(os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv'))



# ==================================================================================
#
# The below function collates the performance metrics for the parameter estimation
# simulations, prints summaries of the results and saves the results as csv files.
#
# ----------------------------------------------------------------------------------
#
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - OutDir: The output directory.
# - nsim: Number of simulations to be collated.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def performanceTables(desInd, OutDir, nsim=1000):

    # Make row indices
    row = ['sim'+str(i) for i in range(1,nsim+1)]

    # Make column indices
    col = ['FS','fFS','SFS','fSFS','cSFS','lmer']

    #-----------------------------------------------------------------------------
    # Work out timing stats
    #-----------------------------------------------------------------------------

    # Make timing table
    timesTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    timesTable = timesTable.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the times
        simTimes = results_table.loc['Time','FS':]

        # Add them to the table
        timesTable.loc['sim'+str(simInd),:]=simTimes

    # Save computation times to csv file
    timesTable.to_csv(os.path.join(OutDir,'timesTable.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of computation times') 
    print(timesTable.describe().to_string())

    #-----------------------------------------------------------------------------
    # Work out number of iteration stats
    #-----------------------------------------------------------------------------

    # Make timing table
    nitTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    nitTable = nitTable.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the times
        simNIT = results_table.loc['nit','FS':]

        # Add them to the table
        nitTable.loc['sim'+str(simInd),:]=simNIT

    # Save number of iterations to csv file
    nitTable.to_csv(os.path.join(OutDir,'nitTable.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of number of iterations')
    print(nitTable.describe().to_string())

    #-----------------------------------------------------------------------------
    # Work out log-likelihood stats
    #-----------------------------------------------------------------------------

    # Make timing table
    llhTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    llhTable = nitTable.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the log-likelihoods
        simllh = results_table.loc['llh','FS':]

        # Add them to the table
        llhTable.loc['sim'+str(simInd),:]=simllh

    # Save log likelihoods to csv file
    llhTable.to_csv(os.path.join(OutDir,'llhTable.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of maximized log-likelihoods')
    print(llhTable.describe().to_string())


# ==================================================================================
#
# The below function collates the MAE and MRD metrics for the parameter estimation 
# simulations, prints summaries of the results and saves the results as csv files.
#
# ----------------------------------------------------------------------------------
#
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - OutDir: The output directory.
# - nsim: Number of simulations to be collated.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def differenceMetrics(desInd, OutDir, nsim=1000):

    # Make row indices
    row = ['sim'+str(i) for i in range(1,nsim+1)]

    # Make column indices
    col = ['FS','fFS','SFS','fSFS','cSFS','lmer']

    #-----------------------------------------------------------------------------
    # Work out absolute difference metrics for lmer
    #-----------------------------------------------------------------------------

    # Make difference tables
    diffTableBetas = pd.DataFrame(index=row, columns=col)
    diffTableVar = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    diffTableBetas = diffTableBetas.apply(pd.to_numeric)
    diffTableVar = diffTableVar.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the betas
        simBetas = results_table.loc['beta1':'beta5',:]

        if desInd==1:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D1,3',:]
        if desInd==2:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D2,3',:]
        if desInd==3:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D3,3',:]

        # Work out the maximum absolute errors for betas
        maxAbsErrBetas = (simBetas.sub(simBetas['lmer'], axis=0)).abs().max()

        # Work out the maximum absolute errors for sigma2D
        if desInd==1:
            maxAbsErrVar = (simVar.sub(simVar['lmer'], axis=0)).abs().max()
        if desInd==2:
            maxAbsErrVar = (simVar.sub(simVar['lmer'], axis=0)).abs().max()
        if desInd==3:
            maxAbsErrVar = (simVar.sub(simVar['lmer'], axis=0)).abs().max()
            
        # Add them to the tables
        diffTableBetas.loc['sim'+str(simInd),:]=maxAbsErrBetas
        diffTableVar.loc['sim'+str(simInd),:]=maxAbsErrVar

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MAE values for beta estimates (compared to lmer)')
    print(diffTableBetas.describe().to_string())
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MAE values for variance estimates (compared to lmer)')
    print(diffTableVar.describe().to_string())

    # Save MAE values for lmer to csv
    diffTableVar.to_csv(os.path.join(OutDir,'diffTableVar_lmer_abs.csv'))
    diffTableBetas.to_csv(os.path.join(OutDir,'diffTableBetas_lmer_abs.csv'))

    #-----------------------------------------------------------------------------
    # Work out absolute difference metrics for Truth
    #-----------------------------------------------------------------------------

    # Make difference tables
    diffTableBetas = pd.DataFrame(index=row, columns=col)
    diffTableVar = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    diffTableBetas = diffTableBetas.apply(pd.to_numeric)
    diffTableVar = diffTableVar.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the betas
        simBetas = results_table.loc['beta1':'beta5',:]

        if desInd==1:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D1,3',:]
        if desInd==2:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D2,3',:]
        if desInd==3:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D3,3',:]

        # Work out the maximum absolute errors for betas
        maxAbsErrBetas = (simBetas.sub(simBetas['Truth'], axis=0)).abs().max()

        # Work out the maximum absolute errors for sigma2D
        if desInd==1:
            maxAbsErrVar = (simVar.sub(simVar['Truth'], axis=0)).abs().max()
        if desInd==2:
            maxAbsErrVar = (simVar.sub(simVar['Truth'], axis=0)).abs().max()
        if desInd==3:
            maxAbsErrVar = (simVar.sub(simVar['Truth'], axis=0)).abs().max()
            
        # Add them to the tables
        diffTableBetas.loc['sim'+str(simInd),:]=maxAbsErrBetas
        diffTableVar.loc['sim'+str(simInd),:]=maxAbsErrVar

    # Save MAE values for truth to csv
    diffTableVar.to_csv(os.path.join(OutDir,'diffTableVar_truth_abs.csv'))
    diffTableBetas.to_csv(os.path.join(OutDir,'diffTableBetas_truth_abs.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MAE values for beta estimates (compared to truth)')    
    print(diffTableBetas.describe().to_string())
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MAE values for variance estimates (compared to truth)') 
    print(diffTableVar.describe().to_string())

    #-----------------------------------------------------------------------------
    # Work out relative difference metrics for lmer
    #-----------------------------------------------------------------------------

    # Make difference tables
    diffTableBetas = pd.DataFrame(index=row, columns=col)
    diffTableVar = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    diffTableBetas = diffTableBetas.apply(pd.to_numeric)
    diffTableVar = diffTableVar.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the betas
        simBetas = results_table.loc['beta1':'beta5',:]

        if desInd==1:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D1,3',:]
        if desInd==2:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D2,3',:]
        if desInd==3:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D3,3',:]

        # Work out the maximum relative differences for betas
        maxRelDiffBetas = (simBetas.sub(simBetas['lmer'], axis=0)).abs().div(simBetas.add(results_table.loc['beta1':'beta5','lmer'],axis=0)/2).max()

        # Work out the maximum relative differences for sigma2D
        if desInd==1:
            maxRelDiffVar = (simVar.sub(simVar['lmer'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D1,3','lmer'],axis=0)/2).max()
        if desInd==2:
            maxRelDiffVar = (simVar.sub(simVar['lmer'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D2,3','lmer'],axis=0)/2).max()
        if desInd==3:
            maxRelDiffVar = (simVar.sub(simVar['lmer'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D3,3','lmer'],axis=0)/2).max()
            
        # Add them to the tables
        diffTableBetas.loc['sim'+str(simInd),:]=maxRelDiffBetas
        diffTableVar.loc['sim'+str(simInd),:]=maxRelDiffVar

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MRD values for beta estimates (compared to lmer)') 
    print(diffTableBetas.describe().to_string())
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MRD values for variance estimates (compared to lmer)') 
    print(diffTableVar.describe().to_string())

    # Save MRD values for lmer to csv
    diffTableVar.to_csv(os.path.join(OutDir,'diffTableVar_lmer_rel.csv'))
    diffTableBetas.to_csv(os.path.join(OutDir,'diffTableBetas_lmer_rel.csv'))

    #-----------------------------------------------------------------------------
    # Work out relative difference metrics for Truth
    #-----------------------------------------------------------------------------

    # Make difference tables
    diffTableBetas = pd.DataFrame(index=row, columns=col)
    diffTableVar = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the table is numeric
    diffTableBetas = diffTableBetas.apply(pd.to_numeric)
    diffTableVar = diffTableVar.apply(pd.to_numeric)

    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the betas
        simBetas = results_table.loc['beta1':'beta5',:]

        if desInd==1:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D1,3',:]
        if desInd==2:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D2,3',:]
        if desInd==3:
            # Get the variance components
            simVar = results_table.loc['sigma2*D1,1':'sigma2*D3,3',:]

        # Work out the maximum relative differences for betas
        maxRelDiffBetas = (simBetas.sub(simBetas['Truth'], axis=0)).abs().div(simBetas.add(results_table.loc['beta1':'beta5','Truth'],axis=0)/2).dropna().max()

        # Work out the maximum relative differences for sigma2D
        if desInd==1:
            maxRelDiffVar = (simVar.sub(simVar['Truth'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D1,3','Truth'],axis=0)/2).dropna().max()
        if desInd==2:
            maxRelDiffVar = (simVar.sub(simVar['Truth'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D1,3','Truth'],axis=0)/2).dropna().max()
        if desInd==3:
            maxRelDiffVar = (simVar.sub(simVar['Truth'], axis=0)).abs().div(simVar.add(results_table.loc['sigma2*D1,1':'sigma2*D1,3','Truth'],axis=0)/2).dropna().max()
            
        # Add them to the tables
        diffTableBetas.loc['sim'+str(simInd),:]=maxRelDiffBetas
        diffTableVar.loc['sim'+str(simInd),:]=maxRelDiffVar

    # Save MRD values for truth to csv
    diffTableVar.to_csv(os.path.join(OutDir,'diffTableVar_truth_rel.csv'))
    diffTableBetas.to_csv(os.path.join(OutDir,'diffTableBetas_truth_rel.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MRD values for beta estimates (compared to truth)') 
    print(diffTableBetas.describe().to_string())
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of MRD values for variance estimates (compared to truth)') 
    print(diffTableVar.describe().to_string())


# ==================================================================================
#
# The below function generates a ground truth degrees of freedom estimate for a 
# given model.
#
# ----------------------------------------------------------------------------------
#
# - X: The fixed effects design matrix.
# - Z: The random effects design matrix.
# - beta: The true fixed effects parameters to be used for simulation.
# - sigma2: The true fixed effects variance to be used for simulation.
# - D: The true random effects covariance matrix to be used for simulation.
# - L: The contrast vector specifying which contrast we wish to estimate the degrees 
#      of freedom for.
# - nlevels: A vector containing the number of levels for each factor,
#            e.g. `nlevels=[3,4]` would mean the first factor has 3
#            levels and the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - tol: Convergence tolerance for the parameter estimation method.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def groundTruth_TDF(X, Z, beta, sigma2, D, L, nlevels, nraneffs, tol):

    # Required product matrices
    XtX = X.transpose() @ X
    XtZ = X.transpose() @ Z
    ZtZ = Z.transpose() @ Z

    # Inverse of (I+Z'ZD) multiplied by D
    DinvIplusZtZD =  forceSym2D(np.linalg.solve(np.eye(ZtZ.shape[0]) + D @ ZtZ, D))

    # Get the true variance of LB
    True_varLB = get_varLB2D(L, XtX, XtZ, DinvIplusZtZD, sigma2)

    # Get the variance of the estimated variance of LB using the 3D code
    var_est_varLB = get_VarhatLB2D(X, Z, beta, sigma2, D, L, nlevels, nraneffs, tol)

    # Get ground truth degrees of freedom
    v = 2*(True_varLB**2)/var_est_varLB

    # Return result
    return(v)


# ==================================================================================
#
# The below function estimates the variance of Var(LB) empirically. It takes the 
# following inputs.
#
# ----------------------------------------------------------------------------------
#
# - X: The fixed effects design matrix.
# - Z: The random effects design matrix.
# - beta: The true fixed effects parameters to be used for simulation.
# - sigma2: The true fixed effects variance to be used for simulation.
# - D: The true random effects covariance matrix to be used for simulation.
# - L: The contrast vector specifying which contrast we wish to estimate the degrees 
#      of freedom for.
# - nlevels: A vector containing the number of levels for each factor,
#            e.g. `nlevels=[3,4]` would mean the first factor has 3
#            levels and the second factor has 4 levels.
# - nraneffs: A vector containing the number of random effects for each
#             factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#             random effects and the second factor has 1 random effect.
# - tol: Convergence tolerance for the parameter estimation method.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def get_VarhatLB2D(X, Z, beta, sigma2, D, L, nlevels, nraneffs, tol):

    # Work out dimensions
    n = X.shape[0]
    p = X.shape[1]
    q = Z.shape[1]
    qu = np.sum(nraneffs*(nraneffs+1)//2)

    # Reshape to 3D dimensions
    X = X.reshape((1,n,p))
    Z = Z.reshape((1,n,q))
    beta = beta.reshape((1,p,1))
    D = D.reshape((1,q,q))

    # New epsilon based on 1000 simulations
    epsilon = np.random.randn(1000, n, 1)

    # Work out cholesky of D
    Dhalf = np.linalg.cholesky(D)

    # New b based on 1000 simulations
    b = Dhalf @ np.random.randn(1000,q,1)

    # New Y based on 1000 simulations
    Y = X @ beta + Z @ b + epsilon

    # Delete b, epsilon, D, beta and sigma^2
    del b, epsilon, D, beta, sigma2

    # Calulcate product matrices
    XtX = X.transpose(0,2,1) @ X
    XtY = X.transpose(0,2,1) @ Y
    XtZ = X.transpose(0,2,1) @ Z
    YtX = Y.transpose(0,2,1) @ X
    YtY = Y.transpose(0,2,1) @ Y
    YtZ = Y.transpose(0,2,1) @ Z
    ZtX = Z.transpose(0,2,1) @ X
    ZtY = Z.transpose(0,2,1) @ Y
    ZtZ = Z.transpose(0,2,1) @ Z

    # Get parameter vector
    paramVec = fSFS3D(XtX, XtY, ZtX, ZtY, ZtZ, XtZ, YtZ, YtY, YtX, nlevels, nraneffs, tol,n,reml=True)

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Retrieve beta estimates
    beta = paramVec[:, 0:p]
    
    # Retrieve sigma2 estimates
    sigma2 = paramVec[:,p:(p+1),:]
    
    # Retrieve unique D estimates elements (i.e. [vech(D_1),...vech(D_r)])
    vechD = paramVec[:,(p+1):,:].reshape((1000,qu))
    
    # Reconstruct D estimates
    Ddict = dict()
    # D as a dictionary
    for k in np.arange(len(nraneffs)):
        Ddict[k] = vech2mat3D(paramVec[:,IndsDk[k]:IndsDk[k+1],:])
      
    # Full version of D estimates
    D = getDfromDict3D(Ddict, nraneffs, nlevels)

    # Inverse of (I+Z'ZD) multiplied by D
    DinvIplusZtZD =  forceSym3D(np.linalg.solve(np.eye(q) + D @ ZtZ, D))

    # Get variance of Lbeta estimates
    varLB = get_varLB3D(L, XtX, XtZ, DinvIplusZtZD, sigma2, nraneffs)

    # Estimated variance of varLB
    varofvarLB = np.var(varLB,axis=0)

    # Reshape and return
    return(varofvarLB.reshape((1,1)))


# ==================================================================================
#
# The below function collates the t-statistics, p-values and degrees of freedom 
# estimates for the T-statistic simulations, prints summaries of the results and
# saves the results as csv files.
#
# ----------------------------------------------------------------------------------
#
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - OutDir: The output directory.
# - nsim: Number of simulations to be collated.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def tOutput(desInd, OutDir, nsim=1000):

    # Make row indices
    row = ['sim'+str(i) for i in range(1,nsim+1)]

    # Make column indices
    col = ['Truth','FS','lmer']

    #-----------------------------------------------------------------------------
    # Work out timing stats
    #-----------------------------------------------------------------------------

    # Make timing table
    tTable = pd.DataFrame(index=row, columns=col)
    pTable = pd.DataFrame(index=row, columns=col)
    dfTable = pd.DataFrame(index=row, columns=col)

    # Make sure pandas knows the tables are numeric
    tTable = tTable.apply(pd.to_numeric)
    pTable = pTable.apply(pd.to_numeric)
    dfTable = dfTable.apply(pd.to_numeric)

    # Loop through and read in simulations
    for simInd in range(1,nsim+1):
        
        # Name of results file
        results_file = os.path.join(OutDir,'Sim'+str(simInd)+'_Design'+str(desInd)+'_results.csv')

        # Read in results file
        results_table = pd.read_csv(results_file, index_col=0)

        # Get the T, P and df values
        simT = results_table.loc['T',['Truth','FS','lmer']]
        simp = results_table.loc['p',['Truth','FS','lmer']]
        simdf = results_table.loc['swdf',['Truth','FS','lmer']]

        # Add them to the tables
        tTable.loc['sim'+str(simInd),:]=simT
        pTable.loc['sim'+str(simInd),:]=simp
        dfTable.loc['sim'+str(simInd),:]=simdf

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of T statistics') 
    print(tTable.describe().to_string())

    # Save T statistics to csv
    tTable.to_csv(os.path.join(OutDir,'tTable.csv'))

    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of P values') 
    print(pTable.describe().to_string())

    # Save p-values to csv
    pTable.to_csv(os.path.join(OutDir,'pTable.csv'))
    
    # Print summary
    print(' ')
    print('--------------------------------------------------------------------------')
    print(' ')
    print('Summary of degrees of freedom estimates') 
    print(dfTable.describe().to_string())

    # Save degrees of freedom estimates to csv
    dfTable.to_csv(os.path.join(OutDir,'dfTable.csv'))


# ==================================================================================
#
# The below function obtains the T-statistics, p-values and degrees of freedom 
# estimates using the parameter estimates and product matrices, via the Direct-SW
# method. It takes the following inputs.
#
# ----------------------------------------------------------------------------------
#
# - `paramVec`: Final estimates of the parameter vector.
# - `XtX`: X transpose multiplied by X.
# - `XtY`: X transpose multiplied by Y.
# - `XtZ`: X transpose multiplied by Z. 
# - `YtX`: Y transpose multiplied by X.
# - `YtY`: Y transpose multiplied by Y.
# - `YtZ`: Y transpose multiplied by Z.
# - `ZtX`: Z transpose multiplied by X.
# - `ZtY`: Z transpose multiplied by Y.
# - `ZtZ`: Z transpose multiplied by Z.
# - `nraneffs`: A vector containing the number of random effects for each
#               factor, e.g. `nraneffs=[2,1]` would mean the first factor has
#               random effects and the second factor has 1 random effect.
# - `nlevels`: A vector containing the number of levels for each factor, 
#              e.g. `nlevels=[3,4]` would mean the first factor has 3 levels
#              and the second factor has 4 levels.
# - `n`: The number of observations.
#
# ----------------------------------------------------------------------------------
#
# Author: Tom Maullin (06/04/2020)
#
# ==================================================================================
def simT(paramVec, XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ, nraneffs, nlevels, n):

    # Scalar quantities
    p = XtX.shape[1] # (Number of Fixed Effects parameters)
    q = np.sum(nraneffs*nlevels) # (Total number of random effects)
    qu = np.sum(nraneffs*(nraneffs+1)//2) # (Number of unique random effects)

    # Output beta estimate
    beta = paramVec[0:p,:]  
    
    # Output sigma2 estimate
    sigma2 = paramVec[p:(p+1),:]

    # Get unique D elements (i.e. [vech(D_1),...vech(D_r)])
    vechD = paramVec[(p+1):,:]

    # Get the indices in the paramvector corresponding to D matrices
    IndsDk = np.int32(np.cumsum(nraneffs*(nraneffs+1)//2) + p + 1)
    IndsDk = np.insert(IndsDk,0,p+1)

    # Reconstruct D
    Ddict = dict()

    # D as a dictionary
    for k in np.arange(len(nraneffs)):

        Ddict[k] = vech2mat2D(paramVec[IndsDk[k]:IndsDk[k+1],:])
        
    # Matrix version
    D = np.array([])
    for i in np.arange(len(nraneffs)):
        for j in np.arange(nlevels[i]):
            # Add block
            if i == 0 and j == 0:
                D = Ddict[i]
            else:
                D = scipy.linalg.block_diag(D, Ddict[i])

    # Contrast vector (1 in last place 0 elsewhere)
    L = np.zeros(p)
    L[-1] = 1
    L = L.reshape(1,p)

    # Miscellaneous matrix variables
    DinvIplusZtZD = D @ np.linalg.inv(np.eye(q) + ZtZ @ D)
    Zte = ZtY - (ZtX @ beta)
    ete = ssr2D(YtX, YtY, XtX, beta)

    # Get T statistic
    T = get_T2D(L, XtX, XtZ, DinvIplusZtZD, beta, sigma2)

    # Get Satterthwaite estimate of degrees of freedom
    df = get_swdf_T2D(L, D, sigma2, XtX, XtZ, ZtX, ZtZ, n, nlevels, nraneffs)

    # Get p value
    # Do this seperately for >0 and <0 to avoid underflow
    if T < 0:
        Pval = 1-stats.t.cdf(T, df)
    else:
        Pval = stats.t.cdf(-T, df)

    # Return T statistic, P value and degrees of freedom
    return(T,Pval,df)
