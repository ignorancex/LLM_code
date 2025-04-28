import importlib
import setup_nx # your own module, setup.nx.py
import scipy.linalg as LA
import math

importlib.reload(setup_nx)
from setup_nx import *
from sympy import * # includes Matrix object and solve function
import scipy.io # need to load .mat

import my_impedance_funcs as imp
import my_heatmapSetup_funcs as hm


def reduceF(Fbig,nzrows,zcols):
    assert isinstance(Fbig, np.ndarray)
    # remove rows having all zeroes
    F = Fbig.copy()
#    F = F[~np.all(F == 0, axis=1)]
    F = F[nzrows]
    # remove cols having all zeroes
    idx = np.argwhere(np.all(F[..., :] == 0, axis=0))
 #   idx = zcols
    F = np.delete(F, idx, axis=1)
    return F

def assignF(parmObj,Fp,Fq,indicMat): # algo similar to updateStateSpace

    n=int(len(indicMat)/6) # indicMat has 6n rows for all versions
    Hblock1=np.zeros((3*n,3*n))   
    ridx,colidx=np.nonzero(indicMat[0:3*n,0:3*n]) # python indexing goes first to (last-1)          
    for k in range(len(ridx)):
            Hblock1[ridx[k]][colidx[k]] = Fq
    
    Hblock2=np.zeros((3*n,3*n))   

    if parmObj.get_version()==1: # PBC       
        ridx,colidx=np.nonzero(indicMat[0:3*n,0:3*n]) 
        for k in range(len(ridx)):
                Hblock2[ridx[k]][colidx[k]] = Fp

        upper=np.concatenate((Hblock1, np.zeros((3*n,3*n))),axis=1)
        lower=np.concatenate((np.zeros((3*n,3*n)),Hblock2),axis=1)
        F=np.concatenate((upper,lower))  # indicMat is 6n x 6n, [Fq 0 ; 0 Fp]
    
    else: # Droop:        
        ridx,colidx=np.nonzero(indicMat[3*n+1:,0:3*n]) 
        for k in range(len(ridx)):
                Hblock2[ridx[k]][colidx[k]] = Fp
        F=np.concatenate((Hblock1,Hblock2),axis=0) # indicMat is now 6n x 3n, [Fq Fp]'

    #print("Size of F=",F.shape)
    return F

def assignF_v2(f_ub_table,n): # algo similar to updateStateSpace
    # create one F matrix from parm space
    # indicMat has diagonal 3x3 blocks where an APNP is
    F=np.zeros((6*n,6*n)) # make numpy matrix that will hold values not symbolic var
    candFset=np.zeros((1,len(f_ub_table)))
    percentExplore=np.array([]) # number of cols will be = number of nonzero F ele

    # load MATLAB .mat with chebyshev ball data
    matfile = scipy.io.loadmat('mycheby.mat') # dict with var names as the keys
    #print(matfile.keys())
    chebyC=Matrix(matfile['chebyC'])
    chebyR=Matrix(matfile['chebyR'])
    #print('chebyC size is ',chebyC.shape)
    assert(len(chebyC)==len(f_ub_table))
    
    for k in range(0,len(f_ub_table)): # for each actuator
        # f_ub_table has format  [APNP_number indicMat_row indicMat_col f_lb f_ub]
        fub=f_ub_table[k][4] # 4 for 5th col, which has upper bound
        #print('fub=',fub)
        i=int(f_ub_table[k][1]) # i from nonzero f_ij ele
        j=int(f_ub_table[k][2]) # j from nonzero f_ij ele

#         #probs=[0.05, 0.1, 0.1, 0.25, 0.45, 0.05, 0.0, 0.0, 0.0, 0.0] # probability distr, elements must sum to 1
#         probs=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] # probability distr, elements must sum to 1
#         #probs=[0, 0, 0, 0, 0.05, 0.1, 0.1, 0.25, 0.45, 0.05] # probability distr, elements must sum to 1
#         fval=np.random.choice(np.linspace(0,fub,len(probs)), p=probs) # take sample from backwards weibul, 45% chance it's 90% of fub
#         #print('fval=',fval)
                    
       # sample according to gaussians around the chebyCenter
        sigma =chebyR*2
        mu=chebyC[k]
        fval=np.random.normal(mu, sigma,1) # mu,sigma,nsamp
        percentExplore=np.append(percentExplore,100*(fval-mu)/mu) # save how far sample is from chebyC
        candFset[0,k]=max([fval, 0]) # save fvals into vec, replace with zero if negative
        # assume all blocks are 3ph for now, will correct later in this funcc
        #print('(i,j)=',i,' and ',j)
        F[i,j]=fval
        
    print('percentage change of chebyC=',percentExplore)

    return F,candFset
   
def assignF_v3(nzrow_dict,sample_starts,std_devs,n): # algo similar to updateStateSpace
    # create one F matrix from parm space
    # nzrow_dict is a dict whos keys are the nonzero rows of indicMat, and who's values are the nonzero elements of each nonzero col
    # sample_starts is sx1 vector of starting values for f elements
    # std_dev is sx1 vector of standard deviations for gaussian sampling from the sample_starts

    #print('std_devs=',std_devs)
    #print('sample_starts=',sample_starts)


    F=np.zeros((6*n,6*n)) # make numpy matrix that will hold values not symbolic var
    candFset=np.zeros((1,len(nzrow_dict.keys())))
    percentExplore=np.array([]) # number of cols will be = number of nonzero F ele
    
    count=0
    zcols=[]
    #print('nzrow_dict.keys()=',nzrow_dict.keys(),flush=True)
    for i in nzrow_dict.keys(): # for each nonzero row of indicMat
        # sample according to gaussians around the sample_start
        sigma =std_devs[count]
        mu=sample_starts[count]
        for j in nzrow_dict[i]: # make all nz elements in each F row sampled from same distribution
            zcols.append([x for x in range(len(F)) if x not in nzrow_dict[i]])

            lb,ub=0.01,100 # sample until fval is in [lb ub] range
            for k in range(100): # resample up to 100 times to get
                fval=np.random.normal(mu, sigma,1) # mu,sigma,nsamp
                if fval>lb and fval<ub:
                    break
                if k==99:
                    raise Exception("could not sample a fval in [lb ub] range")
            
            percentExplore=np.append(percentExplore,100*(fval-mu)/(mu+0.00001)) # save how far sample is from chebyC
            candFset[0,count]=fval # save fvals into vec, replace with zero if negative
            # assume all blocks are 3ph for now, will correct later in this funcc
            #print('F(j,k)=',j,' and ',k)
            F[i,j]=fval

        count+=1
        
#     mystr='gaussian variance making percentExplore too high; max='+str(max(np.absolute(percentExplore)))
#     assert max(np.absolute(percentExplore))<1500,mystr
    #print('percentage change from starting point =',percentExplore,flush=True)


    return F,candFset,list(nzrow_dict.keys()),zcols
        
# version 1, workaround
def computeFParamSpace_v1(feeder, act_locs, perf_nodes):
    # Compute (Fp,Fq) ranges as a func of impedance paths between act nodes, perf nodes, and substation
    gamma=[0.08, 0.35]
    Fplist,Fqlist= (np.empty((0,1)) for i in range(2))
    for i in len(act_locs):
        Fqlist[i]=i
        Fplist[i]=i
        
    Fq_lb=np.min(Fqlist)
    Fp_lb=np.min(Fplist)
    return Fq_lb,Fp_lb

#version 2, correct
def computeFParamSpace_v2(parmObj,feeder, act_locs, perf_nodes,R,X,depths,node_index_Map, file_name):
    # Compute (Fq,Fp) ranges as a func of impedance paths between act nodes, perf nodes, and substation
    if '13NFbalanced' in file_name:
        c=np.array([0.412,0.857]) # (q,p) tuned for 13NF based on data from feas configs
    elif '123NF' in file_name:
        #c=np.array([0.3,0.45])  # setting on 9/2/20
        #c=np.array([0.5,0.7]) # place_max_coloc getting not-great results from this
        if parmObj.get_version()==1:  
            c=np.array([0.3,0.45])  # setting back to this on 2/28/21
        else: # droop case
            c=np.array([0.7,1.1])  # setting this on 3/7/21
    elif 'PL0001' in file_name: 
        c=np.array([0.32,0.65])
    elif 'oaklandJ' in file_name:
        c=np.array([0.3,0.45])
    else:
        print('error: c not assigned because couldnt find load file in folder')

        
    avgSens_dvdq,avgSens_ddeldp= (np.empty((0,1)) for i in range(2))
    i=0
    for act in act_locs: # for each (perf,act) pair in current config you're evaluating
        #print(node_index_Map)
        perf=perf_nodes[i] #index of act node
        i=i+1 # increment to update perf node
        # indexing  a[start:stop] --> items start through stop-1
        act_idx=node_index_Map[act] # index of act node
        perf_idx=node_index_Map[perf] 

        #print('evaluating act at ',act,', perf at ',perf)

        Zgood=np.empty((3,3))
        Zgood=(R[act_idx*3:(act_idx*3+3),perf_idx*3:(perf_idx*3+3)])/2+(X[act_idx*3:(act_idx*3+3),perf_idx*3:(perf_idx*3+3)])/2*1j # 3x3 matrix        
        Z_toSubst=imp.get_total_impedance_from_substation(feeder,act,depths)
        Z_actperf=imp.get_common_node_impedance_two_buses(feeder, act, perf, depths)

        Zbad1=np.subtract(Z_toSubst,Zgood) # >=0
        Zbad2=np.subtract(Z_actperf,Zbad1) # >=0
        #print('Zbad1=',np.around(Zbad1,2))
        #print('Zbad2=',np.around(Zbad2,2))
        
        # deratings should range 0 to 1 for each 3x3 elements
        der1=np.ones(3)-np.divide(Zbad1,Zgood+Zbad1) # derating for actuator not colocated
        der2=np.ones(3)-np.divide(Zbad2,Zgood+Zbad2) # derating for perf node not on samepath to substation as act
        #print('der1=',der1)
        #print('der2=',der2)
        mainPath=Zgood
        sensEst_dvdq=np.multiply(np.multiply(der1,der2),mainPath)
        sensEst_ddeldp=np.multiply(np.multiply(der1,der2),mainPath)

        avg_dvdq=sensEst_dvdq.mean() # converts 3x3 to scalar complex for each act-perf pair
        avg_ddeldp=sensEst_ddeldp.mean()
        
        if np.isnan(avg_dvdq) or np.isnan(avg_ddeldp):
            print('act=',act)  
            sys.exit('Error: avg_dvdq or avg_ddeldp are NaN')  
            
        avgSens_dvdq=np.append(avgSens_dvdq,avg_dvdq)
        avgSens_ddeldp=np.append(avgSens_ddeldp,avg_ddeldp)
        #print('avgdvdq=',avgSens_dvdq)
        #print('avgddeldp=',avgSens_ddeldp)
        
    numact=len(act_locs)
    # avgSens_dvdq is list of scalar complex vals
    zpath_var1=np.var(np.absolute(avgSens_dvdq)) # if z to sub varies a lot per act, more stable so dec p and q
    zpath_var2=np.var(np.absolute(avgSens_ddeldp))
    print('zpath_multiplier1=',np.around(zpath_var1,3))
        
    m=lambda var,nact: 1+(nact-1)*np.exp(-120*var) # lambda func, for high variance m=1, for small variance m=numact
    q=np.mean(np.absolute(avgSens_dvdq))*m(numact,zpath_var1) # take avg across abs value of perf-act pair sensitivities
    p=np.mean(np.absolute(avgSens_ddeldp))*m(numact,zpath_var1)
    #print('q=',q) # print when tuning c1 and c2
    #print('(1/q)*c[0]=',(1/q)*c[0])

    try: 
        # added a floor kgain of (0.05,0.1) due to realism of kgain settings. Result is all controllers will have a limit to # DERs they can place
        Fq_ub=max((1/q)*c[0],0.05) # lower bound on kgains to sample
        Fp_ub=max((1/p)*c[1],0.1)
    except: 
        print('error: q or p = 0, avgSens are=',avgSens_dvdq)

    return Fq_ub,Fp_ub

def computeFParamSpace_v3(B_nonmat,indicMat,indicMat_table):
    B = Matrix(B_nonmat)# convert to matrix opject so can do B.multiply(Fsymb) later
    n=int(len(indicMat)/6) # indicMat always has 6n rows
    row,col=np.nonzero(indicMat) # find indices of nonzero f ele
    #print('row=',row,' and col=',col)
    # if apply gdisc cond to every nonzero ele in F
    f_loc=indicMat_table[:,1:3] # get last 2 cols
    #np.concatenate((np.transpose([row]),np.transpose([col])),axis=1)
    # if only apply gdic cond to one f ele per 3ph block: 
    #f_loc=np.concatenate((np.transpose([row[::3]]),np.transpose([col[::3]])),axis=1) # only get index of first 1 in diagonal 3x3 block     
   
    numact=len(np.unique(f_loc[:,0])) # if one 3ph actuator, numact=3
    numperf=len(np.unique(f_loc[:,1]))      
    num_fele=len(f_loc) 
    print('num_fele=',num_fele)

    # initialize c,r, and bigBox_range
    c=np.zeros((6*n,1))
    c=Matrix(c) # convert to matrix obj so can have symbolic ele
    r=np.zeros((6*n,1))
    r=Matrix(r) 

    # main function call:
    bigBox_range=np.around(det_Gdisc_Franges(indicMat_table, n, B, c, r),2) # round to two decimal places

    return bigBox_range
    
def computeFParamSpace_v4(indicMat,indicMat_table,B):
    
    nzrow={} 
    print('len(indicMat[0]=',len(indicMat[0]))
    for i in range(indicMat.shape[0]): # across rows
        if not (indicMat[i,:]==np.zeros((1,len(indicMat)))).all(): # if not a row of zeros
            foo=np.where(indicMat[i,:]!=0)
            #print('indices not zero are:',foo,flush=True)
            nzrow[i]=foo # key=row of indicMat that is nz, value=indices that are nz
         #   print(foo)
    print('finished making nzrow dictionary in computeFParamSpace_v4...')
    # nzrow is a dict whos keys are the nonzero rows of indicMat, and who's values are the indices of the nonzero elements in each nonzero row

    
    y=np.count_nonzero(indicMat!=0) # number of nonzero f elements 
    print('y=',y)
    sample_starts=[]
    count=0
    print('there are ',len(nzrow.keys()), ' sets of gaussian mu parameters:')
    for j in nzrow.keys(): # across cols of B that are assoc with nz rows of F
        count+=1
        #print('making sample # ',count,'...',flush=True)
        Bcol=B[:,j]
        sample_start=det_sampleStart(Bcol,y)
        sample_starts.append(sample_start)            
        
    f_ub_table=indicMat_table # dont add anything to it
    return nzrow,sample_starts,f_ub_table

def det_sampleStart(Bcol,y): # runs for each Gdisc (each row of Hsub)
    # col_idx is index of Hsub that Bcol is in
    # y is number of nonzero ele in Fsub
    assert(np.count_nonzero(Bcol==0)==0) # make sure no elements of Bcol are zero, otherwise will get divide by zero error
    bmax=max(np.absolute(Bcol))
    mu=(2/y)*(1/bmax)
    #print('bmax=',bmax)
    #print('mu=',np.round(mu,3))   
    
    return mu

# helper func in computeFParamSpace_v3
# def multBF(B,i,j,n):
#     # returns B*F faster than if did B.multiply(Fsymb)
#     # uses how the only nonzero ele of F is at Fij, so only one col of BF is nonzero
#     Bcol=B[:,i]
#     var('f') # declare symbolic
#     BF=zeros(6*n) # faster than creating array and converting to matrix object
#     BF[:,j]=Bcol*f
#     return BF

# helper func in computeFParamSpace_v3, calls det_Gdisc and solve_Gdisc_conds
def det_Gdisc_Franges(indicMat_table, n, B, c, r):
# for each row of matrix BF, determine Gdisc conditions in terms of radius and center
# reduces B and F matrix before applying gdisc conditions
# solve gdisc conditions to compute range on F gains, store in bigBox
    f_loc=indicMat_table[:,1:3] # get last 2 cols
    num_fele=len(f_loc) 
    var('f') # declare symbolic
 
    bigBox_range= np.concatenate((indicMat_table, np.zeros((len(indicMat_table),2))), axis=1) # add 2 columns to end
    
#     for k in range(num_fele): # apply gdisc conditions for each nonzero f ele, each iteration updates row of bigBox_range
#         print('working with ',k,'_th nonzero F ele')
#         Fsymb=zeros(6*n) # faster than creating array and converting to matrix object
#         #num_col_indicMat=len(indicMat[0]) # number of cols in indicMat
#         injNode=f_loc[k][0]
#         sensNode=f_loc[k][1]
#         if injNode<3*n and sensNode>3*n: 
#             raise Exception("cant choose f_loc in upper right of F, f_loc=["+str(injNode)+" "+str(sensNode)+"]")
#         else:
#             Fsymb[injNode,sensNode]=f
#             #c[k]=myBF[sensNode,sensNode] # f_ij has gdisc center at BF_{jj} because BF's jth col is nonzero
#         myBF=multBF(B,injNode,sensNode,n)
#         nnz_idx=0 # initialize index of nonzero center, should only have one because look at one f ele at a time

#         c,r,nnz_idx=det_Gdisc(c,r,myBF,nnz_idx,n) 
#         print('solving Gdisc conditions...')
#         bigBox_range=solve_Gdisc_conds(c,r,nnz_idx,k,bigBox_range) # updates bigBox range with a new range
    
    # print('bigBox_range=',bigBox_range)
    return bigBox_range
    
# helper func in computeFParamSpace_v3
def det_Gdisc(c,r,myBF,nnz_idx,n):
# for each row of matrix BF, determine radius and center of gdisk
# BF should be full size, not reduced; if reduced, rdc_BF[i,i] has no meaning
    for i in range(6*n): # one gdisc for each row of BF (pxp)
        c[i]=myBF[i,i] # index a Matrix object with [row,col] whereas np array with [row][col]
        if c[i]!=0 and nnz_idx==0: 
            nnz_idx=i # should only ever be one because there is one f ele
        sum=0
        
        for j in range(6*n): # for each col of F, F is axp
            #sum=sum+myBF_noD[i,j]
            if i!=j:
                sum=sum+np.absolute(myBF[i,j]) # row sum
            #print(sum)
        r[i]=sum
    return c,r,nnz_idx

# # helper func in computeFParamSpace_v3
# def solve_Gdisc_conds(c,r,nnz_idx,k,bigBox_range): # for each Gdisc, update one row of bigBox_range
#     # must pass in bixBox_range so can update it across calls of this func
#     # modify indicMat_table by adding 2 columns to end, for [flb fub] computed from gdisc cond
#
#     #print('r=',r)
#     #print('c=',c)
#     cond1=c[nnz_idx]+r[nnz_idx]-2 # <0, c+r<2
#     print('cond1=',cond1)
#     cond2=-c[nnz_idx]+r[nnz_idx] # <0, c-r>0
#     print('cond2=',cond2)
#     cond3=c[nnz_idx] # =0, c>0
#     #print('cond3=',cond3)
#
#     sol1=solve(cond1,dict=True) # eqn is set to zero and solved
#     sol2=solve(cond2,dict=True)
#     print('sol1=',sol1)
#     print('sol2=',sol2)
#
#     if sol1 and sol2: # if both give solns
#         if sol1[0][f]<sol2[0][f]:
#             bigBox_range[k][3:5]=[sol1[0][f], sol2[0][f]] # row of bigBox_range
#         else:
#             bigBox_range[k][3:5]=[sol2[0][f], sol1[0][f]]
#         #print('range=',bigBox_range[k][:])
#     else:
#         print('No solution to conditions')
#
#
#     return bigBox_range

#-------------------------------------------------------------------------------------------------------------
    
def eval_Fmat(parmObj,CLmat,domeig_mags): # items we populate for each F tried
    ssError_no_contract,eigs_outside_circle,Fstable=0,0,0 # saves which checks on F fail

    eigs,evecs=LA.eig(CLmat) # closed loop eigenvalues
    eigMags=np.absolute(eigs)

        
    # find dominant eval, regardles of whether the system is stable or not
    eigMags_lst=eigMags.tolist()
    mag_domeig=max(eigMags_lst) 
        
        
#     For version 2 (droop), check that evals are within unit circle,
#     AND vss for vdbc1 and vdbc2 are in 5% range
    if parmObj.get_version()==2: # volt-var and volt-watt control      
      #  bool2=True # temporarily removing from droop
        
        delV=0.0535 # if dbc causes this change in v, want to ensure that all new steady states are within 0.05
        dbc_vec1=delV*np.ones((len(CLmat), 1)) 
        delvss1=np.dot(LA.inv(np.identity(len(CLmat))-CLmat),dbc_vec1) # inv(I-BF)*dbc_vec

        # identify which nodes have delvss=dbc_vec, and throw those out
        counter=0
        delvss1_cleaned=[]
        for v in delvss1:
            if delV!=v: # basically only substation node voltages are unaffected by kgains, so we throw those out
                delvss1_cleaned.append(v)
            else:
                counter+=1

        #print('delvss1 (min,max,median)=(',np.around(np.amin(delvss1_cleaned),5),',',np.around(np.amax(delvss1_cleaned),5),',',np.around(np.median(delvss1_cleaned),5),')')
        #print('num of immovable 3-ph nodes voltages=',counter/3)
        bool2=all(item <0.05 for item in delvss1_cleaned) # check that the remaining are all under 0.05
    else:
        bool2=True # PBC case
   
    
    if all(np.around(eigMags,decimals=6)<=1):
        # require that all evals=1 have null space full of base
        # evecs (no generalized evecs)
        tol=0.0005
        num1evals=sum(np.absolute(eigMags-1)<tol) # numel(find(abs(abs(eigs)-1)<tol))   
        #num1evals=sum(np.absolute(eigs)==1) # numel(find(abs(abs(eigs)-1)<tol))   
        Y = LA.null_space(CLmat-1*np.eye(len(CLmat))) #null(CLmat-eval*eye(size(CLmat,1))); % Y is orthonorm basis matrix
        dimNull=len(Y[0]) # number of cols
        #print('dimNull=',dimNull,' and num1evals=',num1evals)
       
        # find dominant eval for case of stable system
        eigMags_lst=eigMags.tolist()
        eigMags_lst.sort() # convert to list, then sort vector from smallest to largest
        #print('eigMags=',eigMags_lst[:5],',...,',eigMags_lst[-5:]) # print smallest and largest 5 eigMags
        for i in range(num1evals): # remove the k largest eigs, where k=num1evals
            eigMags_lst.pop() # removes last item
        mag_domeig=max(eigMags_lst) 
        
        if bool2 and (dimNull==num1evals):                    
            #print('Found feas F')
            Fstable=1 # means F is stabilizing 

        else: # if F not feas, print in which way it is
            if not bool2: # save which check failed
                ssError_no_contract=1

            else: # nullity condition doesnt hold
                eigs_outside_circle=1

    else: # outside unit circle
        eigs_outside_circle=1

    domeig_mags=np.append(domeig_mags,mag_domeig)
    val=np.sum(eigMags[np.where(eigMags > 1)])
    return val,ssError_no_contract,eigs_outside_circle,domeig_mags,Fstable 
    
    
def detControlMatExistence(parmObj, feeder, A, B, indicMat,indicMat_table,act_locs,perf_nodes,node_index_map,depths,file_name):
#def detControlMatExistence(feeder, act_locs, perf_nodes,A,B,R,X,indicMat):
    n=int(len(indicMat)/6) # indicMat is 6n x 6n
    print('indicMat size is ',n,flush=True)

    #sample_way='heuristic' # you choose
    sample_way='new-heuristic' # you choose
    
    # Initialize arrays, will be populated in loops
    feas=False # boolean
    numfeas,myCosts,domeig_mags= (np.empty((0,1)) for i in range(3))
    dataFull, data_zero = (np.array([]) for i in range(2)) # # [Fp Fq myCost] for each config
    CLmat=np.empty((6*n,6*n))
    eigs_outside_circle,ssError_no_contract=0,0
    print('evaluating kgains sampled from parm space...',flush=True)
    numsamp =20 # todo change '5' back to 20

    #-------------------------------------------------
    if sample_way=='old-heuristic':
        # Heuristically compute sample space
        R,X=hm.createRXmatrices_3ph(feeder, node_index_map,depths,file_name) # need for computing parm space ranges
        Fq_ub,Fp_ub=computeFParamSpace_v2(parmObj,feeder, act_locs, perf_nodes,R,X,depths,node_index_map,file_name)
        print('Fp_range=',Fp_ub,' and Fq_range=',Fq_ub)
        assert(math.sqrt(numsamp) ** 2 == numsamp) # checks that numsamp is a perfect square
        Fq_range=np.linspace(0.0001, Fq_ub, round(sqrt(numsamp)))
        Fp_range=np.linspace(0.0001, Fp_ub, round(sqrt(numsamp)))  
        
        feasFs=np.array([], dtype=np.int64).reshape(0,2)
        
        # heuristic iter
        for Fq in Fq_range:
            for Fp in Fp_range:
                if not(Fq==0 or Fp==0): # skip iteration if either zero
                    #print("(Fp,Fq)=",Fp,",",Fq,")")
                    if np.isnan(Fp) or np.isnan(Fq):
                        sys.exit('Error: Fq or Fp are NaN')

                    F=assignF(parmObj,Fp,Fq,indicMat)
                    #print('sizeA=',B.shape)
                    CLmat=A-np.dot(B,F) # CLmat=A-BF
                    candFset=np.zeros((1,2))
                    candFset[0][0],candFset[0][1]=Fq,Fp # save fvals into vec
                    val,ssErr_bool,ecirc_bool,domeig_mags,Fstable=eval_Fmat(parmObj,CLmat,domeig_mags) # evaluate a single F matrix

                    if Fstable:
                        print('found stabilizing F!')
                        feasFs=np.append(feasFs,candFset,axis=0) # if Fstable, add Fset to new row of feasFs
                    eigs_outside_circle+=ecirc_bool
                    ssError_no_contract+=ssErr_bool
                    myCosts=np.append(myCosts,[[val]],axis=0) # temp
   
    #-------------------------------------------------
    elif sample_way=='new-heuristic':  # new-heuristic way to compute the sample space
        print('in new-heuristic route',flush=True)

        # todo: why do you need to call this line every iteration?
        nzrow_dict,sample_starts,f_ub_table=computeFParamSpace_v4(indicMat,indicMat_table,B) # format is [APNP_number indicMat_row indicMat_col f_lb f_ub]
        # create f_ub_table, formatted as [act_name perf_name indicMat_row indicMat_col f_lb f_ub]
 #       f_ub_table=computeFParamSpace_v3(B, indicMat,indicMat_table) # format is [APNP_number indicMat_row indicMat_col f_lb f_ub]

        str_arr=np.array([], dtype=np.int64).reshape(0,2)
        for idx in f_ub_table[:,0]: # pull APNP idx from first col of f_ub_table
            str_arr=np.append(str_arr,[np.array([act_locs[int(idx)], perf_nodes[int(idx)]])],axis=0) # adds 2x1 col for each APNP pair
        #print('parm space table =\n',np.append(str_arr,f_ub_table[:,1:],axis=1),' << formated as [act_name perf_name indicMat_row indicMat_col f_lb f_ub]',flush=True)

        #------------------------------------------
        
        prev=10 # starting min domeig to compare to
        step0=0.07 # original step size
        step=step0 # initalize sigma step size
        sigma=0.3  # arbitrary starting point for sigma
        sigma_vec,min_domeig_vec,tol,sig_try=[],[],0.001,10 # todo change '3' back to 10
        for sig_count in range(sig_try): # try this many different sigma, and take the case with the lowest domeig 
            sigma_vec.append(sigma)
            min_domeig_vec.append(prev)

            print('------------------------')
            print('sig iteration ',sig_count+1,'; sigma=',sigma)
            print('step=',step)
            sigma=sigma+step
            std_devs=[sigma]*len(sample_starts)
            domeig_mags,stable_domeig_mags=[],[]
            feasFs=np.array([], dtype=np.int64).reshape(0,len(f_ub_table)) # number of cols will be = number of nonzero F ele
            feasFmats=[] # list of the best Fmats

#          #--------------------------
            for k in range(numsamp): # numsamp is number of F matrices to try
            #    F,candFset=assignF_v2(f_ub_table,n) # design F matrix
                F,candFset,nzrows,zcols=assignF_v3(nzrow_dict,sample_starts,std_devs,n) # design F matrix
                CLmat=A-np.dot(B,F) # CLmat=A-BF
                val,ssErr_bool,ecirc_bool,domeig_mags,Fstable=eval_Fmat(parmObj,CLmat,domeig_mags)  # evaluate a single F matrix
                #print('cand F set=',candFset,flush=True)
                # domeig_mags is added to for every iteration k
                
                
                if Fstable:
                    #print('found stabilizing F!')
                    feasFs=np.append(feasFs,candFset,axis=0) # if Fstable, add Fset to new row of feasFs
                    feasFmats.append(F)
                    stable_domeig_mags=np.append(stable_domeig_mags,domeig_mags[-1]) # latest entry is current domeig

                eigs_outside_circle+=ecirc_bool
                ssError_no_contract+=ssErr_bool
                myCosts=np.append(myCosts,[[val]],axis=0) # temp
            #------------------
            percent_feas=len(feasFs)/numsamp
            print(100*percent_feas,' percent of Fs were feas')
            curr=min(domeig_mags)
            print('(curr,prev) domeig=(',np.round(curr,4),',',np.round(prev,4),')')

            if sig_count==0: # initialized
                save_lst=[domeig_mags,stable_domeig_mags,feasFs,feasFmats,sigma,nzrows,zcols] # save items assoc with best sigma case
            if sig_count>0.3*sig_try and np.absolute(curr-prev)<tol: # when min_domeig stops changing so much
                print('----domeig has converged across sigmas----')
                print('case1')
                break
            elif percent_feas<0.1 or curr>1 or curr>=prev: # if not enough stable Fs to choose from
                sigma=sigma-2*step
                step=step*0.5 # reduce step size
                print('case3')
            else: # curr<prev:
                prev=curr
                save_lst=[domeig_mags,stable_domeig_mags,feasFs,feasFmats,sigma,nzrows,zcols] # save items assoc with best sigma case
                print('new min_domeig=',curr)
                print('case4')

            if np.absolute(step) < np.absolute(step0 * 0.1):  # if step has gotten too small, change direction and reset step size to 50% of original
                step = -0.5 * np.sign(step) * step0  # negate the step
                print('changing step direction..')
        
        # fig, (ax1, ax2) = plt.subplots(2)
        # ax1.plot(sigma_vec, c='red', lw=2,label='sigma')
        # ax1.legend(loc='upper left')
        # ax2.plot(min_domeig_vec, c='green',lw=2,label='min dominant eig')
        # plt.ylim([0.99*min(min_domeig_vec), 1.01*max(min_domeig_vec)])
        # ax2.legend(loc='upper left')
        # plt.show()
        # plt.grid()


        #-------------------------------------------------
    else:
        raise Exception("unrecognized string value for sample_way")
        
# [for both heuristic and non-heuristic] gather result
    [domeig_mags,stable_domeig_mags,feasFs,feasFmats,sigma,nzrows,zcols]=save_lst[:]
    min_domeig_mag=min(domeig_mags)
    numfeas=np.append(numfeas,[[len(feasFs)]],axis=0) # number of rows
    #print('feasFs=',np.around(feasFs,3))
    percent_feas=len(feasFs)/numsamp
    print('percent feas=',np.around(percent_feas,3),flush=True)

    print('smallest dom eig mag=',min_domeig_mag,flush=True)
    # histogram(domeig_mags)
    #print('domeig_mags=',domeig_mags)
    if len(stable_domeig_mags)>1:
        plt.hist([round(num, 3) for num in stable_domeig_mags], density=True, bins=15) # round to nearest hundredth
        plt.ylabel('mag of dominant eig')
        mystr='different stable Fs for sigma='+str(sigma)
        plt.xlabel(mystr);
        

   # if feas==True:
    if percent_feas>0.1: # arbitrary
        print("Config good!",flush=True)
        print('feasF shape=',feasFs.shape)
        print('domeig_mags shape=',domeig_mags.shape)

        idx=np.argmin(stable_domeig_mags)
        bestF_asvec=feasFs[idx][:] # the set of F ele that results in the most stable dominant eval
        Fbig=feasFmats[idx] # the F matrix that is best
        bestF_asmat=reduceF(Fbig,nzrows,zcols)
        #print("Best F is (Fp Fq)=",bestF) # typically really tiny, not interesting to print
        feas=True
        

    else:
        bestF_asvec=float("NaN")
        bestF_asmat = float("NaN")
        feas=False
        print("No F found for config --> bad config",flush=True)
        #print("Unit circle fails=",eigs_outside_circle,', ss_error contraction fails=',ssError_no_contract)

    num_act=np.count_nonzero(indicMat)/2
       
    # return feas,feasFs,num_act,numfeas,numTried
    return feas,feasFs,percent_feas,numsamp,num_act,bestF_asvec,bestF_asmat,indicMat,min_domeig_mag

def eval_config_F(parmObj,bestFparm,indicMat,feeder,depths,node_index_map):
                    # NEED FIX
    # takes in particular F mat +indicMat and evaluates whether (A-BF) is stable; if so feas=True
    Fp=bestFparm[0]
    Fq=bestFparm[1]
    F=assignF_v2(parmObj,Fp,Fq,indicMat)
    
    n=int(len(F)/6) # indicMat and F is 6n x 6n
    A, B = hm.setupStateSpace(parmObj,feeder,n,node_index_map,depths)

    # simpler version of detControlMatExistence:
    # Initialize arrays, will be populated by eval_Fmat function
    domeig_mags= (np.empty((0,1)) for i in range(1))
    feasFs=np.array([], dtype=np.int64).reshape(0,2)
    ssError_no_contract,eigs_outside_circle=0,0 # will save which checks on F fail

    CLmat=np.empty((6*n,6*n))
    CLmat=A-np.dot(B,F) # CLmat=A-BF
    val,ssError_no_contract,eigs_outside_circle,domeig_mags,feasFs=eval_Fmat(parmObj,CLmat,domeig_mags)

    # feasFs=0 if F unstable, feasFs=1 if F is stable
    print('len feasFs=',len(feasFs))
    if len(feasFs)>=1:
        feas=True
    else:
        feas=False

    print('Actuator configuration has enough stability margin') if feas else print('Actuator configuration does not have enough stability margin')
    return feas