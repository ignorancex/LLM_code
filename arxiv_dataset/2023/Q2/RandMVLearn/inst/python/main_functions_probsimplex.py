#@title python function
import torch
import random #useful for setting seed
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from torch.autograd import Variable
from joblib import Parallel, delayed
import itertools
torch.manual_seed(12345)
# Set default tensor type
torch.set_default_tensor_type(torch.DoubleTensor)


#use this code, works well
#Simplex Projection
def simplex_proj(mytheta):
    #mytheta is current direction. Want to project this direction onto the probability simplex
    #mytheta should be  a vector
    mytheta=torch.unsqueeze(mytheta,1)
    nVars,ncols=mytheta.size()
    #nVars=mytheta.size()
    if ncols==1:
        mytheta=torch.squeeze(mytheta)
    mytheta_sort,indices=torch.sort(mytheta, descending=True)
    rho2=torch.div( torch.cumsum(mytheta_sort,dim=0)-1 , torch.arange(1,nVars+1))
    max_rho2,indices=torch.max(rho2,0)
    mylambda=torch.div( torch.cumsum( mytheta_sort[ torch.arange(0,indices+1)  ] ,dim=0) -1, indices+1)
    mylambda2=mylambda[indices]
    theta_proj=torch.max(mytheta- torch.ones(nVars)*mylambda2,  torch.zeros(nVars)     )
    theta_proj=torch.unsqueeze(theta_proj,1) #creates a nVars x 1 tensor
    theta_proj[torch.abs(theta_proj)<0.000001]=0
    return theta_proj



def mysvdt(X):
    u0,s0,v0=torch.svd(X,some=True)
    mySVt=torch.matmul(u0, torch.t(v0))
    return mySVt,s0

def myfastinv(X):
    u,s,v=np.linalg.svd(np.matmul(np.transpose(X),X),full_matrices=True)
    uinvs=np.matmul(u, np.diag(np.power(s, -1))) #seems we have to multiply two numbers at a time
    myinv=np.matmul(uinvs, np.transpose(v))
    myinvtorch = torch.from_numpy(myinv)  #converts from numpy to pytorch
    return myinvtorch

def indicator_matrix(Y):
  nc=len(torch.unique(Y)) #number of classes
  n,_=Y.size()
  indvector=torch.zeros(n,nc).type(torch.LongTensor)
  return indvector.scatter_(1,Y.type(torch.LongTensor),1) #set columns as one if row i has class j (j columns)


def myObjective(Z,Y,G,beta,U,myoptscore,lambda_d,M,outcometype,gamma,groups,grpweights,myrho,myeta):
  #assumes Y comes in as 0,1,...
  n,p=Y.size()
  n,k=G.size()
  nc=len(torch.unique(Y))
  res1=0
  if outcometype=='binary':
    if nc==2:

        Yind=indicator_matrix(Y)
        myoptscore=torch.tensor([torch.sqrt(torch.sum(Yind[:,1])/torch.sum(Yind[:,0])),-torch.sqrt(torch.sum(Yind[:,0])/torch.sum(Yind[:,1]))])

        Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), torch.unsqueeze(myoptscore,dim=1))
        res1 += (0.5/n)*torch.norm(Ytilde - torch.matmul(G, beta), 'fro')**2
    elif nc >2: #technically works with n>=2
        #print(nc)
        H=torch.zeros(nc, nc-1)
        Yind=indicator_matrix(Y)
        for l in range(nc-1):
            if l==0:
                sl=torch.sum(Yind[:,0])
                sl1=torch.sum(Yind[:,0:(l+2)])
            else:
                sl=torch.sum(Yind[:,0:(l)])
                sl1=torch.sum(Yind[:,0:(l+1)])
            lterm=((n*torch.sum(Yind[:,l+1]))**(0.5))*( (sl*sl1)**(-0.5))
            secondterm=(-(n*sl)**(0.5))*((torch.sum(Yind[:,l+1])*sl1)**(-0.5))
            for j in range(nc):#since upperbound is exclusive
                if j==l:
                    H[j,l]=lterm
                elif j==l+1:
                    H[j,l]=secondterm
        q,r=torch.linalg.qr(H)
        H=q
        Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), H)
        res1 += (0.5/n)*torch.norm(Ytilde - torch.matmul(G, beta), 'fro')**2
  elif outcometype=='continuous':
    res1 += (0.5/n)*torch.norm(Y - torch.matmul(G, beta), 'fro')**2


  res2=0
  gpLY=0
  nG=len(groups)
  for d in range(M):
    Ud=U[d]
    res2 += (0.5/n)*torch.norm(G - torch.matmul(Z[d], Ud), 'fro')**2 + (0.5)*lambda_d[d]*torch.norm(Ud, 'fro')**2
    mysum=0
    grpweightsd=grpweights[d]
    groupsd=groups[d]
    gammad=gamma[d]
    for g in range(nG):
      mysum=mysum + grpweightsd[g]*torch.norm(gammad[groupsd[g]],p=2)
      gpLY=gpLY+myeta[d]*myrho[d]*torch.norm(gammad,p=1)+(1-myeta[d])*myrho[d]*mysum

  res=res1 + res2 + gpLY
  return res

def myloss(Z,Y,G,beta,U,myoptscore,lambda_d,M,outcometype):
  #assumes Y comes in as 0,1,...
  n,p=Y.size()
  n,k=G.size()
  nc=len(torch.unique(Y))
  #print('nc is',nc)
  res1=0
  if outcometype=='binary':
    if nc==2:

        Yind=indicator_matrix(Y)
        myoptscore=torch.tensor([torch.sqrt(torch.sum(Yind[:,1])/torch.sum(Yind[:,0])),-torch.sqrt(torch.sum(Yind[:,0])/torch.sum(Yind[:,1]))])

        Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), torch.unsqueeze(myoptscore,dim=1))
        res1 += (0.5/n)*torch.norm(Ytilde - torch.matmul(G, beta), 'fro')**2
    elif nc >2: #technically works with n>=2
        H=torch.zeros(nc, nc-1)
        Yind=indicator_matrix(Y)
        for l in range(nc-1):
            if l==0:
                sl=torch.sum(Yind[:,0])
                sl1=torch.sum(Yind[:,0:(l+2)])
            else:
                sl=torch.sum(Yind[:,0:(l)])
                sl1=torch.sum(Yind[:,0:(l+1)])
            lterm=((n*torch.sum(Yind[:,l+1]))**(0.5))*( (sl*sl1)**(-0.5))
            secondterm=(-(n*sl)**(0.5))*((torch.sum(Yind[:,l+1])*sl1)**(-0.5))
            for j in range(nc):#since upperbound is exclusive
                if j==l:
                    H[j,l]=lterm
                elif j==l+1:
                    H[j,l]=secondterm
        q,r=torch.linalg.qr(H)
        H=q
        Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), H)
        res1 += (0.5/n)*torch.norm(Ytilde - torch.matmul(G, beta), 'fro')**2
  elif outcometype=='continuous':

    res1 += (0.5/n)*torch.norm(Y - torch.matmul(G, beta), 'fro')**2
    secondterm=torch.matmul(G, beta)

  res2=0
  for d in range(M):
    Ud=U[d]
    res2 += (0.5/n)*torch.norm(G - torch.matmul(Z[d], Ud), 'fro')**2 + (0.5)*lambda_d[d]*torch.norm(Ud, 'fro')**2


  res=res1 + res2

  return res


def random_features_sparsity(Xdata,num_features,gamma,kernel_param):
  #torch.manual_seed(12345)
  M=len(Xdata)
  Z=list(range(M))
  myb=list(range(M))
  myB=list(range(M))
  myepsilon=list(range(M))
  myomega=list(range(M))
  torch.manual_seed(12345)
  for d in range(M):

    nRows,nVars=Xdata[d].size()
    mygamma=torch.matmul(gamma[d],torch.ones(1,num_features) )  #repeat gamma # of random features times
    myepsilon[d]=torch.randn(nVars,num_features) #simulate epsilon with mean 0 and 1, bandwith used in gamma
    myomega[d]=myepsilon[d] * mygamma

    mypi=torch.Tensor([math.pi])
    myb[d]=2*mypi*torch.rand(1,num_features) #uniform [0, 2pi]

    myB[d]=torch.matmul(torch.ones(nRows,1),myb[d]) #repeat myb nrow times

    myrandfeatures=torch.cos(torch.matmul(Xdata[d],myomega[d]) + myB[d])
    Z[d]=myrandfeatures

  return {
        'Z':Z,
        'myomega':myomega,
        'myb': myb,
        'gamma': gamma,
        'myepsilon':myepsilon,
        'myB': myB
    }

def solve_OptScore_theta(Y,G,theta,myoptscore):
  nc=len(torch.unique(Y))
  #assumes Y is 0,1 for binary
  if nc==2:
    Yind=indicator_matrix(Y)
    myoptscore=torch.tensor([torch.sqrt(torch.sum(Yind[:,1])/torch.sum(Yind[:,0])),-torch.sqrt(torch.sum(Yind[:,0])/torch.sum(Yind[:,1]))])

    Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), torch.unsqueeze(myoptscore,dim=1))

    #check conditions
    Zones=torch.matmul(Yind.type(torch.DoubleTensor),torch.ones(nc,1))

    GtY=torch.matmul(torch.t(G),Ytilde) #G'*Y
    myinv=torch.inverse(torch.matmul(torch.t(G),G))
    #solution for beta
    thetahat=torch.matmul(myinv,GtY)
  elif nc>2:
    #solve for thetahat
    #need to solve for optimal score
    #use that to solve for theta
    H=torch.zeros(nc, nc-1)
    Yind=indicator_matrix(Y)
    n,_=G.size()
    for l in range(nc-1):
        if l==0:
            sl=torch.sum(Yind[:,0])
            sl1=torch.sum(Yind[:,0:(l+2)])
        else:
            sl=torch.sum(Yind[:,0:(l)])
            sl1=torch.sum(Yind[:,0:(l+1)])
        lterm=((n*torch.sum(Yind[:,l+1]))**(0.5))*( (sl*sl1)**(-0.5))
        secondterm=(-(n*sl)**(0.5))*((torch.sum(Yind[:,l+1])*sl1)**(-0.5))
        for j in range(nc):#since upperbound is exclusive
            if j==l:
                H[j,l]=lterm
            elif j==l+1:
                H[j,l]=secondterm
    #print(H)

    #check conditions
    Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), H)
    Zones=torch.matmul(Yind.type(torch.DoubleTensor),torch.ones(nc,1))

    q,r=torch.linalg.qr(H)
    H=q
    Ytilde=torch.matmul(Yind.type(torch.DoubleTensor), H)
    GtY=torch.matmul(torch.t(G),Ytilde) #G'*Y
    myinv=torch.inverse(torch.matmul(torch.t(G),G))
    #solution for beta
    thetahat=torch.matmul(myinv,GtY)
    myoptscore=H

  return {'thetahat': thetahat,
          'OptScore':myoptscore
         }


def solve_theta_continuos(Y,G):
    GtY=torch.matmul(torch.t(G),Y)  #G'*Y
    myinv=torch.inverse(torch.matmul(torch.t(G),G))
    thetahat=torch.matmul(myinv,GtY)
    return thetahat

#Z is a list of rand features for each data view
# solve for individual loadings. output as a list
def solve_ind_loads(G, Z,mylambda,num_features):
    #check dimension of G
    nRows,r=G.size()
    M=len(Z)
    U=list(range(M))
    tildeG=torch.cat(((1/math.sqrt(nRows))*G, torch.zeros([num_features,r]))) # row concatenation
    for d in range(M):
        Zd=Z[d]
        tildeZd=torch.cat( ( (1/math.sqrt(nRows))*Zd,  torch.eye(num_features)*torch.sqrt(torch.tensor(mylambda[d]))))
        ZtG=torch.matmul(torch.t(tildeZd),tildeG)
        #call function for inverse
        ZdtZd=torch.matmul(torch.t(Zd),Zd)
        pd,rr=ZdtZd.size()
        myinv=torch.inverse(ZdtZd)
        #solution for each Ud
        Ud=torch.matmul(myinv,ZtG)
        pd,rr=Ud.size()
        U[d]=Ud
        if (torch.sum(torch.nonzero(torch.sum(Ud*Ud,0)**(0.5)))<rr):
            U[d]=Ud
        else:
            U[d]=Ud
    return U

def solve_joint(Y,mytheta,Z,A,myOmega,M,outcometype,lambda_d,ncomponents):
    #myOmega is K by K-1 matrix
      U=A

      if outcometype=='continuous':
        n,_=Y.size()
        #n=1
        #form tildeY, Y is n x 1 (for single continous outome) or n by q
        M=len(Z)
        r=U[0].size(1) #number of columns
        ZU=[torch.matmul((1/torch.sqrt(torch.tensor(n)))*Z[d],U[d]) for d in range(M)   ]
        tildeY=torch.cat((Y, torch.cat(ZU,1) ),1    )
        tilde_theta=torch.cat(  (mytheta, torch.cat(M*[torch.eye(r)],1) ),1 )
        YBt=(1/2)*torch.matmul(tildeY,torch.t(tilde_theta))
        Ghat2=mysvdt(YBt)
        #Ghat=Ghat2[0]
        Ghat=Ghat2
      elif outcometype=='binary':
        n,_=Y.size()
        #n=1
        Yind=indicator_matrix(Y)
        nc=len(torch.unique(Y))
        myOmega2=myOmega.view(nc,nc-1) #reshape into K by K-1

        YOmega=torch.matmul(Yind.type(torch.DoubleTensor), myOmega2) #NxK times K by K-1
        M=len(Z)
        r=U[0].size(1) #number of columns
        ZU=[torch.matmul((1/torch.sqrt(torch.tensor(n)))*Z[d],U[d]) for d in range(M)   ]

        nvars,ncols=YOmega.size()
        tildeY=torch.cat((YOmega, torch.cat(ZU,1) ),1    )
        tilde_theta=torch.cat(  (mytheta, torch.cat(M*[torch.eye(r)],1) ),1 )
        YBt=(1/2)*torch.matmul(tildeY,torch.t(tilde_theta))
        Ghat2=mysvdt(YBt)
        #Ghat=Ghat2[0]
        Ghat=Ghat2

      return Ghat

#only solves of G in prediction
def solve_joint_predict2(Y,beta,Z,U,M,outcometype,ncomponents):

#if outcometype=='continuous':
  n,_=Y.size()
  Y=torch.zeros_like(Y)
  beta=torch.zeros_like(beta)
  #form tildeY, Y is n x 1 (for single continous outome) or n by q
  M=len(Z)
  r=U[0].size(1) #number of columns
  ZU=[torch.matmul((1/torch.sqrt(torch.tensor(n)))*Z[d],U[d]) for d in range(M)   ]
  tildeY=torch.cat((Y, torch.cat(ZU,1) ),1    )
  tilde_theta=torch.cat(  (beta, torch.cat(M*[torch.eye(r)],1) ),1 )
  YBt2=(1/2)*torch.cat(ZU,1)
  YBt=torch.matmul(YBt2, torch.t(torch.cat(M*[torch.eye(r)],1)))
  Ghat2=mysvdt(YBt)
  Ghat=Ghat2

  return Ghat


def gamma_loss(gamma_temp, Xdata,myB,myepsilon, G,U,num_features):
    #n,num_features=myB.size()
    #U is M by 1
    n,k=G.size()
    nrows,ncols=gamma_temp.size()
    if nrows==1:
      gamma_temp=torch.t(gamma_temp)
    mygamma=torch.matmul(gamma_temp,torch.ones(1,num_features) )
    myomega=myepsilon * mygamma
    myZ=torch.cos(torch.matmul(Xdata,myomega) + myB)
    myloss=(0.5/n)*torch.norm(G-torch.matmul(myZ,U),'fro')**2
    return myloss

def gamma_lossN(gamma_temp, Xdata,myB,myepsilon, G,U,num_features):
    #n,num_features=myB.size()
    #U is M by 1
    n,k=G.size()
    nrows,ncols=gamma_temp.size()
    if nrows==1:
      gamma_temp=torch.t(gamma_temp)
    mygamma=torch.matmul(gamma_temp,torch.ones(1,num_features) )
    myomega=myepsilon * mygamma
    myZ=torch.cos(torch.matmul(Xdata,myomega) + myB)
    myloss=(0.5/n)*torch.norm(G-torch.matmul(myZ,U),'fro')**2
    return myloss

def gamma_func_simplex(gamma_temp, Xdata,myB,myepsilon, G,U,num_features):
    #n,num_features=myB.size()
    #U is M by 1
    n,k=G.size()
    nrows,ncols=gamma_temp.size()
    if nrows==1:
      gamma_temp=torch.t(gamma_temp)
    mygamma=torch.matmul(gamma_temp,torch.ones(1,num_features) )
    myomega=myepsilon * mygamma
    myZ=torch.cos(torch.matmul(Xdata,myomega) + myB)
    gpLY=torch.matmul(torch.t(gamma_temp),torch.ones_like(gamma_temp))
    myobjective=(0.5/n)*torch.norm(G-torch.matmul(myZ,U),'fro')**2 + gpLY
    return myobjective

def kernelparameters(X1,X2,h):
  #h- how many nearest  neighbors
  h=20
  M=len(X1)
  kernel_parm=list(range(M))
  for d in range(M):
    dim1=X1[d].size(0)
    dim2=X2[d].size(0)

    # #euclidean distance
    dist1=list(range(dim1))
    for j in range(dim1):
      #print(j)
      if j < h:
        X3=X1[d][0 : j+h,:]
      elif j >= h:
        X3=X1[d][j-h : j+h,:]
      #print(j,X3.size())
      sqdiff=(X3)**2
      dist1[j]=torch.sum(sqdiff,dim=1)**0.5

    distdata=torch.cat(dist1,dim=0)
    mysigma=(torch.median(distdata))**0.5
    #kernel_parm[d]=torch.median(distdata)
    kernel_parm[d]=mysigma**2
    #kernel_parm[d]=1/(2*mysigma**2) #same as s1 below
    #set sigma (or s) as the median of the euclidean distance amongst the 20 nearest neighbor instances
  return kernel_parm

 #https://papers.nips.cc/paper/2012/file/621bf66ddb7c962aa0d22ac97d69b793-Paper.pdf
#Gaussian kernel functions
#use this to select the number of components
#s1/s2: Gaussian kernel width s for each view for the kernel function
#k(x,y)=exp(-0.5*|x-y|^2/sigma^2).
#same as exp(-s1|x-y|^2) where s1 = 1/(2sigma^2)
def gaussiankernel(X1,X2,eigsplot,kernelpar,TopK):
  dim1=X1.size(0)
  dim2=X2.size(0)

  norms1=torch.sum(X1*X1,dim=1) #X1 should be a vector (variables)
  norms2=torch.sum(X2*X2,dim=1)
  mat1=torch.matmul(torch.unsqueeze(norms1,dim=1),torch.ones(1,dim2))
  mat2=torch.matmul(torch.ones(dim1,1), torch.t(torch.unsqueeze(norms2,dim=1)))
  distmat=mat1+mat2-2*torch.matmul(X1, torch.t(X2))
  kernelpar=kernelpar*kernelpar #need to square for it to be equal with formula
  #and results from sklearn gaussian process
  Kernelmat=torch.exp(-distmat/(2*kernelpar))
  #print(Kernelmat.size()) #n by n matrix


  #Eigen decomposition
  #eigvalues,eigvec=torch.symeig(Kernelmat,eigenvectors=True)
  eigvalues,eigvec=torch.linalg.eigh(Kernelmat)

  topkeig=dim1
  values, indices=torch.topk(eigvalues,topkeig)

  myeig=eigvec[:,indices] #top k eigenvectors
  myeigvalues=eigvalues[indices]
  #print(torch.matmul(torch.t(myeig),myeig)) #eigenvectors are normalized

  #print(myeig)
  #print(myeigvalues.size())


  if eigsplot==True:
    #plot (eigenvalues)
    plt.figure(1)
    plt.scatter(torch.linspace(0,TopK,steps=TopK),myeigvalues[0:TopK],c='r')
    plt.show()

    plt.figure(2)
    imgplot = plt.imshow(Kernelmat)
    plt.colorbar()
    plt.show()

  return {
    'Kernelmat':  Kernelmat,
    'Eigvalues':myeigvalues
  }



#Adapted from Jiuzhou Wang's code from Deep IDA
def Nearest_Centroid_classifier(X_train,X_test,Y_train,Y_test):
    #X_train/test n*p tensor
    #Y_train/test n tensor
    #Assume that the Y_train and Y_test have the same number of groups

    #build the k centroids
    group_index = Y_train.unique()
    centroids = []
    for i in group_index:
        #calculate the centroids for each group based on train data
        centroids.append(torch.mean(X_train[Y_train==i],dim=0,keepdim=False))


    #assign groups for X_test
    labels_test = []
    for j in X_test:
        distance = []
        for k in centroids:
            #the distance (2-norm) of a test point to group k
            distance.append(torch.norm(j-k))
        #assign the group label for point j
        labels_test.append(group_index[distance.index(min(distance))])
    labels_test = torch.tensor(labels_test)

    #calculate accuracy for test data
    acc_test = (torch.sum(Y_test==labels_test)).numpy()/len(Y_test)

    #assign groups for X_train
    labels_train = []
    for j in X_train:
        distance = []
        for k in centroids:
            #the distance (2-norm) of a test point to group k
            distance.append(torch.norm(j-k))
        #assign the group label for point j
        labels_train.append(group_index[distance.index(min(distance))])
    labels_train = torch.tensor(labels_train)

    #calculate accuracy for train data
    acc_train = (torch.sum(Y_train==labels_train)).numpy()/len(Y_train)
    return {
        'predictedYtest':labels_test,
        'predictedYtrain':labels_train,
        'EstErrorTrain':1-acc_train,
        'EstErrorTest':1-acc_test
    }



#this is faster and seem to get better variable selection that above
def AcceProgGradFistaBacktrack(X,gamma,G,U,myB,num_features,myepsilon,max_iter,update_thresh):
  #FISTA with bracktracking from Beck and Teboulle
  #objective history
  ObjHist=torch.zeros(max_iter+1)
  relObjHist=torch.zeros(max_iter+1)
  num_avg_samples=25 #changed this from 10 to 25 on Dec 19 2022
  #gr descent
  #gInit=gamma
  gammaOut=gamma.clone()
  ObjHist[0]=gamma_loss(gammaOut, X,myB,myepsilon, G,U,num_features)
  relObjHist[0]=10000

  #initiate FISTA variables
  tOld, gY, alpha, beta =1, gammaOut.clone(), 100, 0.5
      #accelerated projected gradient descent
  for i in range(1, max_iter+1):
      gammad_Old=gY.clone()
      # gammad_Old.zero_()
      gammad_Old.requires_grad_(True)
      gamma2=gamma_loss(gammad_Old, X,myB,myepsilon, G,U,num_features)
      if gammad_Old.grad is not None:
        gammad_Old.grad.zero_()
      gamma2.backward() #computes gradient
      j=0
      with torch.no_grad():
          while alpha > update_thresh:
            grad_new=gY - (1/alpha)*gammad_Old.grad

      #projection
            mygammanew=simplex_proj(torch.squeeze(grad_new))
            gradDiff=mygammanew - gY
            ObjNew=gamma_loss(mygammanew, X,myB,myepsilon, G,U,num_features)
            ObjOld=gamma_loss(gY, X,myB,myepsilon, G,U,num_features)
            gpLY=torch.matmul(torch.t(mygammanew),torch.ones_like(mygammanew))
            QL=ObjOld + torch.matmul(torch.t(gammad_Old.grad), gradDiff) + (0.5 *alpha  * torch.matmul(torch.t(gradDiff), gradDiff)) + gpLY
            gYOld=gY

            if ObjNew <= QL:
                tNew=0.5 + 0.5*math.sqrt(1 + 4.0 *tOld *tOld)  #eqn 4.2 in FISTA paper
                gY=mygammanew + ((tOld -1 )/tNew) *(mygammanew- gammaOut) #eqn 4.3 in FISTA
                #update titer and gamma
                tOld,gammaOut=tNew,mygammanew.clone()
                break
            else:
                alpha=(2**j)*alpha #L=2^jL_{i-1}
                j=j+1

      loss=gamma_loss(gammaOut, X,myB,myepsilon, G,U,num_features)
      ObjHist[i]=loss.detach()
      reldiff= torch.norm(gammaOut-gYOld,'fro')**2 / torch.norm(gYOld,'fro')**2
      relObj=torch.abs(ObjHist[i]-ObjHist[i-1])/ObjHist[i-1]
      relObjHist[i]=relObj
      #print('rel diff and obj at iter ', i, reldiff, relObj)
      # if ( torch.min(reldiff,relObj) < update_thresh):
      #   print('rel diff and obj at iter conver', i, reldiff, relObj)
      #   break

      if i > num_avg_samples \
        and (torch.mean(ObjHist[i - (num_avg_samples - 1):i]) - ObjHist[i]) < update_thresh:
          #print("update thresh [{}] for accel proj satisfied at interval {}, exiting...".format(update_thresh, i))
          break
      if i > num_avg_samples \
        and (torch.mean(relObjHist[i - (num_avg_samples - 1):i]) - relObjHist[i]) < update_thresh:
          #print("update thresh [{}] for accel proj satisfied at interval {}, exiting...".format(update_thresh, i))
          break

  return gammaOut.detach() #detaches gradient


#main algorithm for nonlinear joint association and prediction
#July 15, 2022
def NSIRAlgorithmFISTA(Xdata, Y, myseed=25, ncomponents=0,num_features=[], outcometype='continuous',kernel_param=[],mylambda=[], max_iter_nsir=500, max_iter_PG=500, update_thresh_nsir=10**-6,update_thresh_PG=10**-6,standardize_Y=False,standardize_X=False):
  torch.manual_seed(seed=myseed)

  if not isinstance(Xdata,list):
        print("Input should be a list of pytorch arrays!")

  #obtain Z
  n,p=Y.size()
  M=len(Xdata)
  nc=len(torch.unique(Y))
  gamma=list(range(M))
  U=list(range(M))
  mycenter_X=list(range(M))
  mystd_X=list(range(M))
  mycenter_Y=list(range(1))
  mystd_Y=list(range(1))
  num_avg_samples=10

  Yold=Y
  XOld=Xdata

  if standardize_Y==True:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Y,dim=0)
      mystd_Y[0]=torch.std(Y,dim=0)
      Y=torch.div(Y-torch.mean(Y,dim=0).repeat(n,1),torch.std(Y,dim=0))
    elif outcometype=='binary':
      Y=Y
  elif standardize_Y==False:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Y,dim=0)
      Y=Y-torch.mean(Y,dim=0)
    elif outcometype=='binary':
      Y=Y

  #if standardize X is true
  if standardize_X==True:
    for d in range(M):
      mymean=torch.mean(Xdata[d],dim=0)
      mystd=torch.std(Xdata[d],dim=0)
      Xdata[d]=torch.div(Xdata[d]-mymean,mystd)
      mycenter_X[d]=mymean
      mystd_X[d]=mystd

  #estimate kernel parameter if empty
  if not kernel_param:
     if n<=1000:
       kernel_param=median_heuristicsbatch(Xdata)
     elif n>1000:
       kernel_param=kernelparameters(Xdata,Xdata,h=20)

  #set mylambda to 1 for all view if empty
  if not mylambda:
     mylambda=[1]*M

  if not num_features:
    n1=Yold.size(0)
    print('n1 is', n1)
    if n1>=1000:
      num_features=int(300)
    else:
      num_features=int(torch.floor(torch.tensor(n1/2)))
      print('num_features is', num_features)

  #estimate number of components if emtpy
  if ncomponents==0:
    ncomponents=chooseK(Xdata, kernel_param,eigsplot=False, TopK=20, threshold = 0.1, verbose=True)
    ncomponents=int(ncomponents)

  #ObjHist=[]
  for d in range(M):
    n,pd=(Xdata[d]).size()
    gamma[d]=torch.ones(pd,1)/kernel_param[d]
    gamma[d]=gamma[d]/torch.sum(gamma[d])
    chec=torch.squeeze(torch.ones(pd,1)/kernel_param[d])
 
  #random features
  myZout=random_features_sparsity(Xdata,num_features,gamma,kernel_param)

  myB=myZout['myB']
  myepsilon=myZout['myepsilon']
  myomega=myZout['myomega']
  myZ=myZout['Z']
  myb=myZout['myb']

  #initialize
  G=torch.rand(Xdata[0].size(0),ncomponents)
  U=[torch.rand(num_features,ncomponents) for d in range(M)]
  U=[U[d]/torch.norm(U[d],'fro') for d in range(M) ]
  q=Y.size(1)
  if outcometype=='continuous':
    mybeta=torch.zeros(ncomponents,q)
    myoptscore=torch.eye(n)
  elif outcometype=='binary':
    myoptscore=torch.rand(nc,nc-1)
    mybeta=torch.rand(ncomponents,nc-1)


  #track objective
  ObjHist=torch.zeros(max_iter_nsir+1)
  ObjHist[0]=myloss(myZ,Y,G,mybeta,U,myoptscore,mylambda,M,outcometype)
  RelObjHist=torch.zeros(max_iter_nsir+1)
  RelObjHist[0]=1

  for i in range(1,max_iter_nsir+1):
    print("iteration", i)
    #new rescaling gamma with other parameters fixed
    gamma_temp=gamma
    for d in range(M):
      gamma[d]=AcceProgGradFistaBacktrack(Xdata[d],gamma_temp[d],G,U[d],myB[d],num_features,myepsilon[d],max_iter_PG,update_thresh_PG)
      nrows,ncols=gamma[d].size()
      if nrows==1:
        gamma[d]=torch.t(gamma[d])
      mygamma=torch.matmul(gamma[d],torch.ones(1,num_features) )   #repeat gamma # of random features times
      myomega[d]=myepsilon[d] * mygamma
      myZ[d]=torch.cos(torch.matmul(Xdata[d],myomega[d]) + myB[d])

    #solve for Ud
    U=solve_ind_loads(G, myZ,mylambda,num_features)
    U=[U[d]/torch.norm(U[d],'fro') for d in range(M)]

    #solve for G, beta and bias
    if outcometype=='binary':
      Ghat=solve_joint(Y,mybeta,myZ,U,myoptscore,M,outcometype,mylambda,ncomponents)
      G=Ghat[0]
    elif outcometype=='continuous':
      Ghat=solve_joint(Y,mybeta,myZ,U,myoptscore,M,outcometype,mylambda,ncomponents)
      G=Ghat[0]
    #solve for beta
    if outcometype=='continuous':
      mybeta=solve_theta_continuos(Y,G)
      myoptscore=torch.eye(n) #this is a placeholder, doesn't get used
    elif outcometype=='binary':
      solOpt=solve_OptScore_theta(Y,G,mybeta,myoptscore)
      mybeta=solOpt['thetahat']
      myoptscore=solOpt['OptScore']

   #keep track of objective
    ObjHist[i]=myloss(myZ,Y,G,mybeta,U,myoptscore,mylambda,M,outcometype)
    relObj=torch.abs(ObjHist[i]-ObjHist[i-1])/(ObjHist[i-1])
    RelObjHist[i]=relObj
    #print('Objective, relobj', i, ObjHist[i],RelObjHist[i])
    # if (relObj < update_thresh_nsir):
    #   print('convergence at iteration',i, relObj)
    #   break

    # if i > num_avg_samples \
    #     and (torch.mean(ObjHist[i - (num_avg_samples - 1):i]) - ObjHist[i]) < update_thresh_nsir:
    #       print("update thresh [{}] for nsir satisfied at interval {}, exiting...".format(update_thresh_nsir, i))
    #       break
    if i > num_avg_samples \
        and (torch.mean(RelObjHist[i - (num_avg_samples - 1):i]) - RelObjHist[i]) < update_thresh_nsir:
          print("update thresh [{}] for randmvlearn satisfied at interval {}, exiting...".format(update_thresh_nsir, i))
          break

  if outcometype=='continuous':
    #print('mybeta',mybeta)
    Fnorm=torch.norm(mybeta, 'fro')
    if Fnorm!=0:
        mybeta=mybeta/torch.norm(mybeta, 'fro')


  gammaAsOnes=list(range(M))
  gamma2=gamma
  for d in range(M):
    gt=gamma2[d]
    gt[torch.abs(gt)<0.00001]=0
    mysel=abs(gt)>0.0
    gammaAsOnes2=torch.zeros_like(gamma2[d])
    gammaAsOnes2[mysel==True]=1.0
    gammaAsOnes[d]=gammaAsOnes2.detach().numpy()
    gamma[d]=gt

  return {
        'Z': myZ,
        #'myomega':myomega,
        'myb': myb,
        'gamma': gamma,
        'myepsilon':myepsilon,
        #'myBmat': myB,
        'Ghat': G.detach(),
        #'EigenvaluesG':Ghat[1].detach(),
        'Ahat': U,
        'thetahat':mybeta.detach(),
        'ObjHist':ObjHist[2:i].detach().numpy(),
        'RelObjHist':RelObjHist[2:i].detach().numpy(),
        'Var_selection':gammaAsOnes,
        #'Y_mean':mycenter_Y,
        #'Y_std':mystd_Y,
        #'X_mean':mycenter_X,
        #'X_std':mystd_X,
        'Xdata':XOld,
        'Y': Yold,
        'kernel_param':kernel_param,
        'ncomponents':ncomponents,
        'num_features':num_features,
        'standardize_X':standardize_X,
        'standardize_Y':standardize_Y,
        'outcometype':outcometype

    }


#tring new prediction with augmented data
def solve_joint_predict2New(Y,beta,Z,Ztrain,U,M,outcometype,ncomponents):

#if outcometype=='continuous':
  ntrain,_=Y.size()
  ntest,_=Z[0].size()
  Yaug=torch.cat((Y, torch.zeros(ntest,1) ),0    )
  n,_=Yaug.size()
  M=len(Z)
  r=U[0].size(1) #number of columns
  ZdAug=list(range(M))
  for d in range(M):
    ZdAug[d]=torch.cat((Ztrain[d],Z[d] ),0    )
  ZU=[torch.matmul((1/torch.sqrt(torch.tensor(n)))*ZdAug[d],U[d]) for d in range(M)   ]

  tildeY=torch.cat((Yaug, torch.cat(ZU,1) ),1    )
  tilde_theta=torch.cat(  (beta, torch.cat(M*[torch.eye(r)],1) ),1 )

  YBt2=(1/2)*torch.cat(ZU,1)
  YBt=torch.matmul(YBt2, torch.t(torch.cat(M*[torch.eye(r)],1)))
  Ghat2train=mysvdt(YBt[0:ntrain,:])
  Ghat2test=mysvdt(YBt[ntrain:n,:])
  Ghattrain=Ghat2train[0]
  Ghattest=Ghat2test[0]
  return{
    'Ghattrain':Ghattrain,
    'Ghattest':Ghattest
  }


def PredictYNew(Ytest,Ytrain,Xtest,Xtrain,Gtrain=[],myb=[],gamma=[],thetahat=[],Ahat=[],myepsilon=[],outcometype=[],num_features=[],standardize_Y=[],standardize_X=[]):
    #need to use testing data to compute Ztest, and then to compute Gtest and use that to obtain predicted Y
    #form Ztest
    Uhat=Ahat
    mybetahat=thetahat
    M= len(Xtest)
    myB=list(range(M))
    myomega=list(range(M))
    myZ=list(range(M))
    myZtrain=list(range(M))
    Xtest_center=list(range(M))
    Xtrain_center=list(range(M))
    ZU=0

    ntest,_=Xtest[0].size()
    ntrain,_=Gtrain.size()

    Ytest_Orig=Ytest
    Xtest_Orig=Xtest

    Xtrain_Orig=Xtrain
    Ytrain_Orig=Ytrain

    if standardize_X==True:
      for d in range(M):
        mX=torch.mean(Xtrain[d],dim=0)
        stdX=torch.std(Xtrain[d],dim=0)
        mycenter=Xtest[d]-mX.repeat(ntest,1)

        #standardize training data
        Xtrain[d]=torch.div(Xtrain[d]-mX,stdX)

        #standardize Xtest with mean and standard deviation from Xtrain
        Xtest[d]=torch.div(Xtest[d]-mX,stdX)

    if outcometype=='continuous':
         Y_mean=torch.mean(Ytrain,dim=0)
         Y_std=torch.std(Ytrain,dim=0)

    for d in range(M):
        nTestRows,pd=Xtest[d].size()
        mygamma=torch.matmul(gamma[d],torch.ones(1,num_features) )   #repeat gamma # of random features times
        myomega[d]=myepsilon[d] * mygamma
        myB[d]=torch.matmul(torch.ones(nTestRows,1),myb[d]) #repeat myb nrow times
        myZ[d]=torch.cos(torch.matmul(Xtest[d],myomega[d]) + myB[d])
        myZtrain[d]=torch.cos(torch.matmul(Xtrain[d],myomega[d]) + torch.matmul(torch.ones(ntrain,1),myb[d]))
        ZU=ZU + torch.matmul(myZ[d],Uhat[d])


    nTestRows,pd=Xtest[0].size()
    pd,ncomponents=Uhat[0].size()

    #calculate training error- don't transform back to original data
    if outcometype=='continuous':

        if standardize_Y==True:

          Ystd=torch.div(Ytrain-Y_mean.repeat(ntrain,1),Y_std)
          Gtrain=solve_joint_predict2(Ytrain,mybetahat,myZtrain,Uhat,M,outcometype,ncomponents)
          GYt=torch.matmul(Gtrain[0],mybetahat)
          TrainError=torch.mean( (GYt-Ystd)**2)

        elif standardize_Y==False: #defaults to centering Y
          Gtrain=solve_joint_predict2(Ytrain,mybetahat,myZtrain,Uhat,M,outcometype,ncomponents)
          Ycentered=Ytrain-Y_mean.repeat(ntrain,1)
          GYt=torch.matmul(Gtrain[0],mybetahat)
          TrainError=torch.mean( (GYt-Ycentered)**2)

    #testing
    if outcometype=='continuous':
      #print('checking')
      Gtest=solve_joint_predict2(Ytest,mybetahat,myZ,Uhat,M,outcometype,ncomponents)
      predictedY=torch.matmul(Gtest[0],mybetahat)
      if standardize_Y==True:
         Yteststd=torch.div(Ytest-Y_mean.repeat(ntest,1),Y_std)
         mse=torch.mean( (predictedY-Yteststd)**2)
         TestError=mse

      elif standardize_Y==False:
         Ytestcentered=Ytest-Y_mean.repeat(ntest,1)
         mse=torch.mean( (predictedY-Ytestcentered)**2)
         TestError=mse

      myPredict=predictedY
      LDAScoreTest=0
      LDAScoreTrain=0
        #check if some columns are zero, set MSE to very large so that corresponding tuning parameters
      ifzero=list(range(M))
      for d in range(M):
          ifzero[d]=torch.sum(abs(gamma[d]))
      if(torch.min(torch.Tensor(ifzero))==0):
          TestError=100000.0
          TrainError=100000.0

    elif outcometype=='binary':
      Gtrain=solve_joint_predict2(Ytrain,mybetahat,myZtrain,Uhat,M,outcometype,ncomponents)
      Ytestn=torch.zeros(Ytest.size(0),1)
      Gtest=solve_joint_predict2(Ytestn,mybetahat,myZ,Uhat,M,outcometype,ncomponents)
      LDAScoreTest=torch.matmul(Gtest[0], mybetahat)
      LDAScoreTrain=torch.matmul(Gtrain[0], mybetahat)
      n,r=LDAScoreTest.size()
      if r >= 1:
            LDAScoreTest=torch.squeeze(LDAScoreTest)
      n,r=LDAScoreTrain.size()
      if r >= 1:
            LDAScoreTrain=torch.squeeze(LDAScoreTrain)
      n,r=Ytrain.size()
      if r >= 1:
            YtrainNew=torch.squeeze(Ytrain)
      n,r=Ytest.size()
      if r >= 1:
            YtestNew=torch.squeeze(Ytest)
      #nearest centroid classification
      myPredict=Nearest_Centroid_classifier(LDAScoreTrain,LDAScoreTest,YtrainNew,YtestNew)
      TestError=myPredict['EstErrorTest']
      TrainError=myPredict['EstErrorTrain']

      ifzero=list(range(M))
      for d in range(M):
          ifzero[d]=torch.sum(abs(gamma[d]))
      if(torch.min(torch.Tensor(ifzero))==0):
          TestError=1.0
          TrainError=1.0

    return {
          'predictedEstimates':myPredict,
          'TestError':TestError,
          'TrainError':TrainError,
          'Gtest':Gtest[0],
          'Gtrain':Gtrain[0],
          'Xtest_standardized':Xtest
        }

def median_heuristicsbatch(X):
  M=len(X)
  kernel_param=list(range(M))
  n=X[0].size(0)
  norm2sq2jj=list()

  nn=int(300)
  T=int(torch.ceil(torch.div(n,nn)))
  mymin=min(n,nn)
  results=torch.zeros(mymin, mymin)
  norm2sq2jj=torch.zeros(T,1)
  for d in range(M):
    for jj in torch.arange(start=1, end=T+1, step=1):
      batch_index=  torch.arange(start=nn*(jj-1), end=min(n,nn*jj), step=1)
      #nn*(jj-1)+1 : torch.min(n,nn*jj)
      Xbatch=X[d][batch_index,:]
      for i in range(Xbatch.size(0)):
        for j in torch.arange(start=0, end=i, step=1):
          mydiff=Xbatch[i,:] - Xbatch[j,:]
          norm2sq=(torch.norm(mydiff,p=2))**2
          results[i,j]=norm2sq

      #print(results)
      norm2sq2=torch.flatten(results) #convert to a vector
      #norm2sq2= torch.stack(results)
      cc=norm2sq2!=0
      norm2sq2b=torch.unsqueeze(norm2sq2,1)
      norm2sq2jj[jj-1,:]=np.median(norm2sq2[norm2sq2!=0])

    #mymedian=torch.mean(torch.tensor(norm2sq2jj))
    mymedian=torch.mean(norm2sq2jj)
    #print(mymedian)

    kernel_param[d]=(mymedian/2)**0.5
  return kernel_param




def chooseK(X, kernelpar, eigsplot=True, TopK=30, threshold = 0.1, verbose=True):
  if threshold <= 0:
      raise Exception("Threshold value must be positive.")

  M = len(X)
  kchoosed=list(range(M))
  Xtrain2=list(range(M))
  Xtrain2=X

  for d in range(M):
    mygauss=gaussiankernel(Xtrain2[d],Xtrain2[d],eigsplot,kernelpar[d],TopK)
    myeigsval=mygauss['Eigvalues']

  # Select K based on percent change in eigenvalues
    calc2 = list()
    for j in range(len(myeigsval)-1):
      calc2.append((myeigsval[j] - myeigsval[j+1])/myeigsval[j])
      print('mychange is ',calc2[j])
      if(calc2[j] < threshold):
        print('mychange is ',calc2[j])
        kchoosed[d] = j+1
        if verbose:
          print("K Based on simple approach using", threshold, "as cut-off:", kchoosed[d], 'for view',d)
        break

  kchoose=torch.min(torch.Tensor(kchoosed)) #min k for common K
  print('K to use is',kchoose)
  return kchoose



#simulate data for coninuous outcome
def generateContData(myseed=1234,n1=200,n2=500,p1=1000,p2=1000,nContVar=1,
    sigmax1=0.1,sigmax2=0.1,sigmay=0.1,ncomponents=3):

  torch.manual_seed(seed=myseed)
  pd=[p1,p2]
  n1=int(n1/2)
  n2=int(n2/2)
  n=n1+n2
  X1=torch.linspace(0.6,2.5,steps=n1)
  Xr1=(X1-1.0)**2
  Xr2=(X1+0.1)**2-2*Xr1


  X2=torch.linspace(0.96,1.67,steps=n2)
  Xr12=(X2-1)**2 + 0.25
  Xr22=(X2+.1)**2-3.5*Xr12


  X1=torch.unsqueeze(X1,dim=1)

  Xclass11=torch.cat((X1,X1),dim=0)
  Xclass12=torch.cat((Xr1,Xr2),dim=0)
  Xclass12=torch.unsqueeze(Xclass12,dim=1)

  Xclass13=torch.matmul(Xclass12,torch.ones(1,p1-1) )

  Xdata11=torch.cat((Xclass11,Xclass13),dim=1)

  che=torch.cat((Xclass11,Xclass12),dim=0)

  d=0

  nonzeros=20
  zrs=pd[d]-nonzeros
  W=torch.cat( (torch.ones(2*n1,nonzeros), torch.zeros(2*n1, zrs)), dim=1)

  Xdata1=Xdata11*W + sigmax1*torch.randn(2*n1,pd[0])

  #########################
  #2nd dataset
  ########################
  nn,_=Xdata1.size()
  Xdata2=5*Xdata1 + sigmax2*torch.randn(nn,pd[1])

  X=[Xdata2, Xdata1]

  ##################
  #Generate G
  ##################

  u,s,v=torch.linalg.svd(Xdata11*W)
  G=u[:,0:ncomponents]
  #print(G[1:10,:])

  nn,_=G.size()

  ##############
  # Generate Y
  ##############
  #generate continuous Y
  # if outcometype=='continuous':
  # bias=0

  beta=torch.Tensor(ncomponents,nContVar).uniform_(0, 1)
  Y=5*torch.matmul(G,beta) + sigmay*torch.randn(nn,nContVar) #this was set to Y=5*G + E in results from Sep 9. But keeping Sep 9 old results.
  cc=torch.linspace(1,nn,steps=nn)
  cc2=torch.unsqueeze(torch.t(cc),dim=1)

  return {
    'X': X,
    'G':G,
    'Y':Y,
}


def generateBinaryData(myseed=1234,n1=200,n2=500,p1=1000,p2=1000,sigmax11=0.1,sigmax12=0.1,sigmax2=0.2):

  torch.manual_seed(seed=myseed)

  n1=int(n1/2)
  n2=int(n2/2)
  pd=[p1,p2]

  #n=n1+n2
  X1=torch.linspace(0.6,2.5,steps=n1)
  Xr1=(X1-1.0)**2
  Xr2=(X1+0.1)**2-2*Xr1


  X2=torch.linspace(0.96,1.67,steps=n2)
  Xr12=(X2-1)**2 + 0.25
  Xr22=(X2+.1)**2-3.5*Xr12

  X1=torch.unsqueeze(X1,dim=1)
  Xclass11=torch.cat((X1,X1),dim=0)
  Xclass12=torch.cat((Xr1,Xr2),dim=0)
  Xclass12=torch.unsqueeze(Xclass12,dim=1)
  #repeat Xclass12 p1- 1 times
  Xclass13=torch.matmul(Xclass12,torch.ones(1,p1-1) )

  Xdata11=torch.cat((Xclass11,Xclass13),dim=1)

  #multiply by W
  #signal=0.1
  d=0
  #nonzeros=math.floor(signal*pd[d])
  nonzeros=20
  zrs=pd[d]-nonzeros
  W=torch.cat( (torch.ones(2*n1,nonzeros), torch.zeros(2*n1, zrs)), dim=1)
  Xdata11=Xdata11*W + sigmax11*torch.randn(2*n1,pd[0])

  ################################
  #Class two
  ################################
  X2=torch.unsqueeze(X2,dim=1)
  # print(X2.size())
  Xclass21=torch.cat((X2,X2),dim=0)
  Xclass22=torch.cat((Xr12,Xr22),dim=0)
  Xclass22=torch.unsqueeze(Xclass22,dim=1)
  #repeat Xclass12 p1- 1 times
  Xclass23=torch.matmul(Xclass22,torch.ones(1,p1-1) )
  Xdata12=torch.cat((Xclass21,Xclass23),dim=1)

  #multiply by W
  d=0
  nonzeros=20
  zrs=pd[d]-nonzeros
  W=torch.cat( (torch.ones(2*n2,nonzeros), torch.zeros(2*n2, zrs)), dim=1)
  Xdata12=Xdata12*W + sigmax12*torch.randn(2*n2,pd[0])

  Xdata1=torch.cat((Xdata11,Xdata12),dim=0)
  nn,_=Xdata1.size()
  plt.figure(3)
  plt.scatter(Xdata1[0:2*n1,0], Xdata1[0:2*n1,1], c = 'b')
  plt.scatter(Xdata1[2*n1:nn,0], Xdata1[2*n1:nn,1], c = 'r')
  plt.show()

  # plt.figure(5)
  # imgplot = plt.imshow(Xdata1)
  # plt.colorbar()
  # plt.show()
  #########################
  #2nd dataset
  ########################
  nn,_=Xdata1.size()
  Xdata2=5*Xdata1 + sigmax2*torch.randn(nn,pd[1])
  plt.figure(4)
  plt.scatter(Xdata2[0:2*n1,0], Xdata2[0:2*n1,1], c = 'b')
  plt.scatter(Xdata2[2*n1:nn,0], Xdata2[2*n1:nn,1], c = 'r')
  plt.show()

  # plt.figure(6)
  # imgplot = plt.imshow(Xdata2)
  # matplotlib.pyplot.box(on=True)
  # plt.colorbar()
  # plt.show()

  X=[Xdata1, Xdata2]

  ##################
  #Generate G
  ##################
  Y=torch.cat([torch.zeros(Xdata11.size(0)),torch.ones(Xdata12.size(0))])
  Y=torch.unsqueeze(Y,dim=1)
  return {
    'X': X,
    'Y':Y
}


####################Group variable selection codes
def SparseIndGroupParallel(kk,Xtrain, Ytrain, myseed, ncomponents,num_features,hasGroupInfo, outcometype,kernel_param,mylambda, lassopenalty_list, myeta,groupsd,max_iter_nsir,
                   max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X,Yvalid,Xvalid):

        nrows=len(lassopenalty_list)
        myvalidMSEsmat=torch.zeros(nrows,2)
        lassopenaltynum=torch.zeros(nrows,1)
        lassopenaltyd=torch.Tensor(lassopenalty_list[kk])
        #temp=lassopenalty_list
        #temp=lassopenalty_list[kk].clone()
        #lassopnealtyd=lassopenalty_list[kk].clone()
        #lassopnealtyd=temp[kk]

        print('working on grid', kk,lassopenaltyd)

        myalg=NSIRAlgorithmFISTASIndandGLasso(Xtrain, Ytrain, myseed, ncomponents,num_features, hasGroupInfo,groupsd,outcometype,
        kernel_param,mylambda, lassopenaltyd, myeta,max_iter_nsir, max_iter_PG,
        update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)

        predict_Y=PredictYNew(Yvalid,Ytrain,Xvalid,Xtrain,myalg['Ghat'],myalg['myb'],myalg['gamma'],myalg['thetahat'],myalg['Ahat'],
                                          myalg['myepsilon'],outcometype,num_features,standardize_Y,standardize_X)


        myvalidMSEsmat=predict_Y['TestError']
        lassopenaltynum=kk

        return myvalidMSEsmat,lassopenaltynum

####For group and individual variable selection

def AcceProgGradFistaSGlassoNew2(X,gamma,G,U,myB,num_features,myrho,eta,grpweights,groups,myepsilon,max_iter,update_thresh):
  #FISTA with bracktracking from Beck and Teboulle
  #objective history
      ObjHist=torch.zeros(max_iter+1)
      relObjHist=torch.zeros(max_iter+1)
      num_avg_samples=15 #changed this from 10 to 25 on Dec 19 2022
      ObjHist[0]=gamma_loss(gamma, X,myB,myepsilon, G,U,num_features)
      relObjHist[0]=10000
      nG=len(groups)
      tvec=torch.zeros(max_iter+2)
      tvec[0]=1
      Lvec=torch.zeros(max_iter+2)
      Lvec[0]=100
      tOldOld=1
      tOld=1
      mygammai=list(range(max_iter+2))
      mygammai[0]=gamma.clone()
      mygammai[1]=gamma.clone()
      for i in range(1, max_iter+1):
          #print('alpha at iteration i', alpha)

          if i==1:
             alphai=(tOldOld-1)/tOld
          else:
            alphai=(tvec[i-2]-1)/tvec[i-1]

          gY=mygammai[i] + alphai*(mygammai[i]- mygammai[i-1])
          gammad_Old=gY.clone()

          gammad_Old.requires_grad_(True)
          gamma2=gamma_loss(gammad_Old, X,myB,myepsilon, G,U,num_features)
          if gammad_Old.grad is not None:
             gammad_Old.grad.zero_()
          gamma2.backward() #computes gradient
          for j in range (0, 100):
            L=(2**j)*Lvec[i-1]
            with torch.no_grad():
                    grad_new=gY - (1/L)*gammad_Old.grad
                    myrho2=myrho #loss going down without division by L; interesting
                    mygammanew=SparseGroupLassoNew(grad_new,myrho2,eta,grpweights,groups)
                    gradDiff=mygammanew - gY
                    #ObjNew=gamma_loss(mygammanew, X,myB,myepsilon, G,U,num_features)
                    ObjNew=gamma_func(mygammanew, X,myB,myepsilon, G,U,num_features,grpweights,groups,myrho2,eta)
                    ObjOld=gamma_loss(gY, X,myB,myepsilon, G,U,num_features)
                    mysum=0
                    for g in range(nG):
                      mysum=mysum + grpweights[g]*torch.norm(mygammanew[groups[g]],p=2)

                    gpLY=eta*myrho2*torch.norm(mygammanew,p=1)+(1-eta)*myrho2*mysum
                    QL=ObjOld + torch.matmul(torch.t(gammad_Old.grad), gradDiff) + (0.5*L  * torch.matmul(torch.t(gradDiff), gradDiff)) + gpLY

                    gYOld=gY

                    if ObjNew <= QL:


                        mygammai[i+1]=mygammanew.clone()
                        Lvec[i]=L

                        break
          tvec[i]=0.5 + 0.5*math.sqrt(1 + 4.0 *tvec[i-1] *tvec[i-1])  #eqn 4.2 in FISTA paper


          if(torch.norm(mygammanew,p=2)!=0):
            gammaOut=mygammanew
          loss=gamma_loss(mygammanew, X,myB,myepsilon, G,U,num_features)
          ObjHist[i]=loss.detach()
          reldiff= torch.norm(mygammanew-gYOld,'fro')**2 / torch.norm(gYOld,'fro')**2
          relObj=torch.abs(ObjHist[i]-ObjHist[i-1])/ObjHist[i-1]
          relObjHist[i]=relObj
          if(torch.sum(abs(mygammanew))==0):
            print('all zeros')
            break


          if i > num_avg_samples \
            and (torch.mean(ObjHist[i - (num_avg_samples - 1):i]) - ObjHist[i]) < update_thresh:
              #print("update thresh [{}] for nsir satisfied at interval {}, exiting...".format(update_thresh, i))
              break

      return mygammanew.detach() #detaches gradient



def NSIRAlgorithmFISTASIndandGLasso(Xdata, Y,myseed, ncomponents,num_features, hasGroupInfo,
          GroupIndices,outcometype,kernel_param,mylambda, myrho, myeta,max_iter_nsir,
          max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X):
  #grpweights is a list of list-eg grpweiths[M][groups]
  #lassopenalty is a list for each View
  torch.manual_seed(seed=myseed)

  if not isinstance(Xdata,list):
        print("Input should be a list of pytorch arrays!")
  #obtain Z
  #obtain Z
  n,p=Y.size()
  M=len(Xdata)
  nc=len(torch.unique(Y))
  gamma=list(range(M))
  U=list(range(M))
  mycenter_X=list(range(M))
  mystd_X=list(range(M))
  mycenter_Y=list(range(1))
  mystd_Y=list(range(1))
  lassopenaltyd=list(range(M))
  num_avg_samples=10
  Yold=Y
  XOld=Xdata


  if standardize_Y==True:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Yold,dim=0)
      mystd_Y[0]=torch.std(Yold,dim=0)
      Y=torch.div(Yold-torch.mean(Yold,dim=0).repeat(n,1),torch.std(Y,dim=0).repeat(n,1))
    elif outcometype=='binary':
      Y=Yold
  elif standardize_Y==False:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Yold,dim=0)
      Y=Yold-torch.mean(Yold,dim=0)
    elif outcometype=='binary':
      Y=Yold


  #if standardize X is true
  if standardize_X==True:
    for d in range(M):
      mymean=torch.mean(XOld[d],dim=0)
      mystd=torch.std(XOld[d],dim=0)
      Xdata[d]=torch.div(XOld[d]-mymean.repeat(XOld[0].size(0),1),mystd.repeat(XOld[0].size(0),1))
      mycenter_X[d]=mymean
      mystd_X[d]=mystd


   #estimate kernel parameter if empty
  # if not kernel_param:
  #    kernel_param=median_heuristicsbatch(Xdata)

  if not kernel_param:
     if n<=1000:
       kernel_param=median_heuristicsbatch(Xdata)
     elif n>1000:
       kernel_param=kernelparameters(Xdata,Xdata,h=20)

  #set mylambda to 1 for all view if empty
  if not mylambda:
     mylambda=[1]*M

  #set myeta to 0.5 if not set
  if not myeta:
     myeta=list(range(M))
     for d in range(M):
       if hasGroupInfo[d]==1:
         myeta[d]=0.5
       elif hasGroupInfo[d]==0:
         myeta[d]=0.0


  if not num_features:
    n1=Yold.size(0)
    print('n1 is', n1)
    if n1>=1000:
      num_features=int(300)
    else:
      num_features=int(torch.floor(torch.tensor(n1/2)))
      print('num_features is', num_features)

  #estimate number of components if emtpy
  if ncomponents==0:
    ncomponents=chooseK(Xdata, kernel_param,eigsplot=False, TopK=20, threshold = 0.1, verbose=True)
    ncomponents=int(ncomponents)

  #get group weights
  gWeight=getWeights(GroupIndices,hasGroupInfo)
  grpweights=gWeight['grpweightsd']
  groups=gWeight['groupsd']
  #print(grpweights)
  #print(groups)



  rhoUpper=list(range(M))
  for d in range(M):
    n,pd=(Xdata[d]).size()
    if hasGroupInfo[d]==1:
        gamma[d]=torch.ones(pd,1)/kernel_param[d]
        ggd=gamma[d]
        for g in range(len(grpweights[d])):
            den=grpweights[d][g]
            num=ggd[groups[d][g].long()]
            uG=num/den
            ggd[groups[d][g].long()]=uG
        gamma[d]=ggd
        #rhoUpper[d]=tunerange(gamma[d],myeta[d],grpweights[d],groups[d])

    elif hasGroupInfo[d]==0:
        gamma[d]=torch.ones(pd,1)/kernel_param[d]
        #rhoUpper[d]=0.0 #just a place holder

  #random features
  myZout=random_features_sparsity(Xdata,num_features,gamma,kernel_param)


  myB=myZout['myB']
  myepsilon=myZout['myepsilon']
  myomega=myZout['myomega']
  myZ=myZout['Z']
  myb=myZout['myb']



  #initialize
#   q,r=torch.linalg.qr(torch.randn(Xdata[0].size(0),ncomponents))
#   G=q

  #G=torch.rand(Xdata[0].size(0),ncomponents)
  q,r=torch.linalg.qr(torch.randn(Xdata[0].size(0),ncomponents))
  G=q

  U=[torch.rand(num_features,ncomponents) for d in range(M)]
  U=[U[d]/torch.norm(U[d],'fro') for d in range(M) ]

  q=Y.size(1)
  if outcometype=='continuous':
    mybeta=torch.zeros(ncomponents,q)
    myoptscore=torch.eye(n)
  elif outcometype=='binary':
    myoptscore=torch.rand(nc,nc-1)
    mybeta=torch.rand(ncomponents,nc-1)

  #track objective
  ObjHist=torch.zeros(max_iter_nsir+1)
  ObjHist[0]=myloss(myZ,Y,G,mybeta,U,myoptscore,mylambda,M,outcometype)

  RelObjHist=torch.zeros(max_iter_nsir+1)
  RelObjHist[0]=1
  ifzero=list(range(M))
  gamma_temp=gamma

  mynonzeros=torch.nonzero(torch.Tensor(hasGroupInfo))
  mynzmat=torch.zeros(mynonzeros.size(0),2) # a zero matrix with number of nonzeros as rows
  #and 2 columns; first column
  try:
    cc=torch.FloatTensor(myrho)
  except:
    cc=torch.Tensor(myrho)
  mynzmat[:,0]=torch.t(cc[mynonzeros]) #extract rho for nonzeros
  mynzmat[:,1]=torch.t(mynonzeros)

  for i in range(1,max_iter_nsir+1):
    print("iteration", i)
    for d in range(M):

          if hasGroupInfo[d]==1:
              myrhonew=mynzmat[mynzmat[:,1]==d,0]
              gamma_temp[d]=gamma[d]
              tic = time.perf_counter()
              gamma[d]=AcceProgGradFistaSGlassoNew2(Xdata[d],gamma_temp[d],G,U[d],myB[d],num_features,myrhonew,myeta[d],grpweights[d],
                                              groups[d],myepsilon[d],max_iter_PG,update_thresh_PG)
              toc=time.perf_counter()
              times1=toc-tic

          elif hasGroupInfo[d]==0:
              gamma[d]=AcceProgGradFistaBacktrack(Xdata[d],gamma_temp[d],G,U[d],myB[d],num_features,
                                                   myepsilon[d],max_iter_PG,update_thresh_PG)

          nrows,ncols=gamma[d].size()
          if nrows==1:
            gamma[d]=torch.t(gamma[d])
          mygamma=torch.matmul(gamma[d],torch.ones(1,num_features) )   #repeat gamma # of random features times
          myomega[d]=myepsilon[d] * mygamma
          myZ[d]=torch.cos(torch.matmul(Xdata[d],myomega[d]) + myB[d])

          ifzero[d]=torch.sum(abs(gamma[d]))

    if(torch.min(torch.Tensor(ifzero))==0):
        print('all zeros in some gamma, consider changing lasso penalty: error with this gamma will be set to a large number')
        break

    #solve for Ud
    U=solve_ind_loads(G, myZ,mylambda,num_features)
    U=[U[d]/torch.norm(U[d],'fro') for d in range(M)]

    #solve for G, beta and bias
    if outcometype=='binary':
      Ghat=solve_joint(Y,mybeta,myZ,U,myoptscore,M,outcometype,mylambda,ncomponents)
      G=Ghat[0]
    elif outcometype=='continuous':
      Ghat=solve_joint(Y,mybeta,myZ,U,myoptscore,M,outcometype,mylambda,ncomponents)
      G=Ghat[0]
    #solve for beta
    if outcometype=='continuous':
      mybeta=solve_theta_continuos(Y,G)
      myoptscore=torch.eye(n) #this is a placeholder, doesn't get used
    elif outcometype=='binary':
      solOpt=solve_OptScore_theta(Y,G,mybeta,myoptscore)
      mybeta=solOpt['thetahat']
      myoptscore=solOpt['OptScore']

   #keep track of objective
    ObjHist[i]=myloss(myZ,Y,G,mybeta,U,myoptscore,mylambda,M,outcometype)
    relObj=torch.abs(ObjHist[i]-ObjHist[i-1])/(ObjHist[i-1])
    RelObjHist[i]=relObj

    if i > num_avg_samples \
        and (torch.mean(ObjHist[i - (num_avg_samples - 1):i]) - ObjHist[i]) < update_thresh_nsir:
          print("update thresh [{}] for nsir satisfied at interval {}, exiting...".format(update_thresh_nsir, i))
          break
    if i > num_avg_samples \
        and (torch.mean(RelObjHist[i - (num_avg_samples - 1):i]) - RelObjHist[i]) < update_thresh_nsir:
          print("update thresh [{}] for nsir satisfied at interval {}, exiting...".format(update_thresh_nsir, i))
          break

  if outcometype=='continuous':
    Fnorm=torch.norm(mybeta, 'fro')
    if Fnorm!=0:
        mybeta=mybeta/torch.norm(mybeta, 'fro')

  gammaAsOnes=list(range(M))
  gamma2=gamma
  for d in range(M):
    gt=gamma2[d].detach()
    gt[torch.abs(gt)<0.00001]=0
    mysel=abs(gt)>0.0
    gammaAsOnes2=torch.zeros_like(gamma2[d].detach())
    gammaAsOnes2[mysel==True]=1.0
    gammaAsOnes[d]=gammaAsOnes2
    gamma[d]=gt

  #calculate group selected
  gammaGroup=GroupSelected(gammaAsOnes,GroupIndices, hasGroupInfo)
  gammaasonesn=list(range(M))
  gammaGroupn=list(range(M))
  for d in range(M):
    gammaasonesn[d]=gammaAsOnes[d].detach().numpy()
    if hasGroupInfo[d]==1:
      gammaGroupn[d]=gammaGroup[d].detach().numpy()
    else:
      gammaGroupn[d]=gammaGroup[d]

  return {
        'Z': myZ,
        'myb': myb,
        'gamma': gamma,
        'myepsilon':myepsilon,
        'Ghat': G.detach(),
        'Ahat': U,
        'thetahat':mybeta.detach(),
        'ObjHist':ObjHist[2:i].detach(),
        'RelObjHist':RelObjHist[2:i].detach(),
        'Var_selection':gammaasonesn,
        'GroupSelection':gammaGroupn,
        'Xdata':XOld,
        'Y': Yold,
        'kernel_param':kernel_param,
        'ncomponents':ncomponents,
        'num_features':num_features,
        'standardize_X':standardize_X,
        'standardize_Y':standardize_Y,
        'outcometype':outcometype,
        'GroupInfo': hasGroupInfo,
        'GroupIndices': GroupIndices,
        'myrho': myrho,
        'myeta':myeta
    }




def SparseGroupLassoNew(gamma,myrho,eta,grpweights,groups):
    nG=len(groups)
    grouplassoproj=gamma.clone()
    for gindex in range(nG):

            uG=gamma[groups[gindex]]
            #lasso projection on group
            pd,_=uG.size()
              #proximal operator for lasso regularizer
            ll3=abs(uG)- myrho
            lassoproj=torch.mul(torch.sign(uG),torch.max(ll3,torch.zeros(pd,1))) #lasso solution
            #check if group is zero
            uGnorm=torch.norm(lassoproj,p=2)
            if uGnorm <= (1-eta)*myrho*grpweights[gindex]:
                grouplassoproj[groups[gindex]]=torch.Tensor(abs(lassoproj)*torch.Tensor([0 ]))
            else:
                grouplassoproj[groups[gindex]]=lassoproj
                gammaIn=grouplassoproj
                uGnormin=1/uGnorm
                ll4=1-(1-eta)*myrho*grpweights[gindex]*uGnormin
                grouplassoproj[groups[gindex]]=torch.Tensor(lassoproj*torch.max(ll4,0)[0])
    return grouplassoproj

def getWeights(GroupIndicies,hasGroupInfo):
  M=len(GroupIndicies)
  grpweightsd=list(range(M))
  groupsd=list(range(M))
  for d in range(M):
    if hasGroupInfo[d]==1:
        temp=GroupIndicies[d].copy()
        MyGroupInfoData=torch.from_numpy(temp)
        nG=len(torch.unique(MyGroupInfoData[:,0])) #number of groups
        groups=list(range(nG))
        gW=list(range(nG))
        for j in range(1,nG+1):
            myg=MyGroupInfoData[MyGroupInfoData[:,0]==j,1]-1
            groups[j-1]=myg.long()
            gW[j-1]=len(MyGroupInfoData[MyGroupInfoData[:,0]==j,1])**0.5

        grpweightsd[d]=gW
        groupsd[d]=groups
    elif hasGroupInfo[d]==0:
        grpweightsd[d]=[]
        groupsd[d]=[]
  return {
    'groupsd': groupsd,
    'grpweightsd':grpweightsd
   }


def gamma_func(gamma_temp, Xdata,myB,myepsilon, G,U,num_features,grpweights,groups,myrho,eta):
    #U is M by 1
    nG=len(groups)
    n,k=G.size()
    nrows,ncols=gamma_temp.size()
    if nrows==1:
      gamma_temp=torch.t(gamma_temp)
    mygamma=torch.matmul(gamma_temp,torch.ones(1,num_features) )
    myomega=myepsilon * mygamma
    myZ=torch.cos(torch.matmul(Xdata,myomega) + myB)
    mysum=0
    for g in range(nG):
      mysum=mysum + grpweights[g]*torch.norm(gamma_temp[groups[g]],p=2)

    gpLY=eta*myrho*torch.norm(gamma_temp,p=1)+(1-eta)*myrho*mysum
    myobjective=(0.5/n)*torch.norm(G-torch.matmul(myZ,U),'fro')**2 + gpLY
    return myobjective


def GroupSelected(gammaAsOnes,GroupIndices, hasGroupInfo):
  M=len(gammaAsOnes)
  GroupSelection=list(range(M))
  for d in range(M):
    gammaAsOnesd=gammaAsOnes[d]
    cc=torch.t(torch.squeeze(gammaAsOnesd))
    cc2=torch.unsqueeze(cc,dim=1)
    if hasGroupInfo[d]==1:
        temp=GroupIndices[d].copy()
        MyGroupInfoData=torch.from_numpy(temp)
        nG=len(torch.unique(MyGroupInfoData[:,0])) #number of groups
        nSelected=torch.zeros(nG,2)
        for j in range(1,nG+1):
            myg=MyGroupInfoData[MyGroupInfoData[:,0]==j,1]-1
            nSelected[j-1,0]=j
            nSelected[j-1,1]=torch.sum(cc2[myg.long()]==1)
        GroupSelection[d]=nSelected
    elif hasGroupInfo[d]==0:
       GroupSelection[d]=0

  return GroupSelection



#################Cross-validation
def chooselassoValid2IndGroup(Xtrain, Ytrain,Xvalid,Yvalid,myseed,ncomponents,num_features,hasGroupInfo,numbercores,outcometype,kernel_param,mylambda, myeta,
                         myrhomin, myrhomax,groupsd,gridmethod,ngrid,max_iter_nsir, max_iter_PG,
                         update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X):
#grplassopenalty - list of d entries for each view
#myrhomin-list of d entries of minimum lassopenalty to use, set to [] if no group information
#myrhomax- list of d entries of maximum lassopenalty to use, set to [] if no group information
  #set defaults
  M=len(Xtrain)
  nofnonzerogrp=torch.nonzero(torch.Tensor(hasGroupInfo)).size(0)
  lassopenalty_listd=list(range(nofnonzerogrp))
  lassomin2=myrhomin
  lassomax2=myrhomax
  nNoGroup=0; #counter for number of views with no group information
  mygroupinfo=torch.nonzero(torch.Tensor(hasGroupInfo))

  for d in range(nofnonzerogrp):
    #if the group is empty, then there's no group information
        if hasGroupInfo[mygroupinfo[d]]==0: #there's no group information
            lassomin2[d]=0
            lassomax2[d]=0
            nNoGroup=nNoGroup+1
        else: #there's group information
            nNoGroup=nNoGroup+0
            if myrhomin[mygroupinfo[d]]==0:
              lassomin2[d]=0.0000001
            else:
              lassomin2[d]=myrhomin[mygroupinfo[d]]

            if myrhomax[mygroupinfo[d]]==0:
              lassomax2[d]=0.00001
            else:
              lassomax2[d]=myrhomax[mygroupinfo[d]]

            lassopenalty_listd[d]=torch.linspace(lassomin2[d],lassomax2[d],ngrid)

  lasso_cat=torch.cat(lassopenalty_listd)
  myPermutations=list(itertools.permutations(lasso_cat, r=mygroupinfo.size(0)))
  myPermutations.sort()
  uniquecomb=torch.Tensor(list(myPermutations for myPermutations,_ in itertools.groupby(myPermutations)))
  if gridmethod=='RandomSearch':
    random.seed(12345)
    ntrialsn=torch.ceil(torch.tensor(0.2*len(uniquecomb))) #keeps 20%
    if ntrialsn <=2:
       ntrials=1
    else:
       ntrials=ntrialsn
    lassopenalty_list=random.sample(list(uniquecomb),int(ntrials))
  elif gridmethod=='GridSearch':
    lassopenalty_list=uniquecomb

  nlist=len(lassopenalty_list)
  #print("lassopenalty_list",lassopenalty_list)

  if numbercores==0: #checks if ncores is empty, then uses half the number of CPU cores
    ncores=int(torch.ceil(torch.Tensor([mp.cpu_count()/2])))
  else:
    ncores=numbercores

  myparallel = Parallel(n_jobs=ncores,prefer="threads",verbose=100, pre_dispatch='1.5*n_jobs')(delayed(SparseIndGroupParallel)(kk,Xtrain, Ytrain,myseed, ncomponents,num_features,
  hasGroupInfo,outcometype,kernel_param,mylambda, lassopenalty_list, myeta, groupsd,max_iter_nsir, max_iter_PG,
  update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X,Yvalid,Xvalid) for kk in range(nlist))

  #obtain lambda corresponding to smallest error
  myout=torch.Tensor(myparallel)
  myvalidMSEs=myout[:,0]
  [minValid, minK]=torch.min(torch.Tensor(myvalidMSEs),dim=0)

  lassopenaltynum=myout[:,1]

  #use the penalty yielding minimum error
  lassopenaltyd=lassopenalty_list[minK]
  print('optimal penalty',lassopenaltyd)
  grplassopenalty=lassopenaltyd
  myalg=NSIRAlgorithmFISTASIndandGLasso(Xtrain, Ytrain, myseed, ncomponents,num_features, hasGroupInfo,groupsd,
  outcometype,kernel_param,mylambda, lassopenaltyd, myeta,max_iter_nsir,
  max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)


  myrho_list=lassopenalty_list

  return myalg, myvalidMSEs,myrho_list,torch.squeeze(torch.Tensor(lassopenaltynum))



def nsrf_cvindgrouplassoNew(Xtrain, Ytrain,myseed,ncomponents,num_features,hasGroupInfo,numbercores,
outcometype,kernel_param,mylambda, myeta, rhoLower, rhoUpper,groupsd,gridmethod,nfolds,ngrid,max_iter_nsir,
max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X):

  torch.manual_seed(seed=myseed)

  if not isinstance(Xtrain,list):
        print("Input should be a list of pytorch arrays!")

  nrow,_= Ytrain.size()
  n,p=Ytrain.size()
  M=len(Xtrain)
  nc=len(torch.unique(Ytrain))
  gamma=list(range(M))
  U=list(range(M))
  mycenter_X=list(range(M))
  mystd_X=list(range(M))
  mycenter_Y=list(range(1))
  mystd_Y=list(range(1))
  lassopenaltyd=list(range(M))
  num_avg_samples=10

  Yold=Ytrain
  XOld=Xtrain


  if standardize_Y==True:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Yold,dim=0)
      mystd_Y[0]=torch.std(Yold,dim=0)
      Y=torch.div(Yold-torch.mean(Yold,dim=0).repeat(n,1),torch.std(Yold,dim=0).repeat(n,1))
    elif outcometype=='binary':
      Y=Yold
  elif standardize_Y==False:
    if outcometype=='continuous':
      mycenter_Y[0]=torch.mean(Yold,dim=0)
      Y=Yold-torch.mean(Yold,dim=0)
    elif outcometype=='binary':
      Y=Yold


  #if standardize X is true
  if standardize_X==True:
    for d in range(M):
      mymean=torch.mean(XOld[d],dim=0)
      mystd=torch.std(XOld[d],dim=0)
      Xtrain[d]=torch.div(XOld[d]-mymean.repeat(XOld[0].size(0),1),mystd.repeat(XOld[0].size(0),1))
      mycenter_X[d]=mymean
      mystd_X[d]=mystd


  # #estimate kernel parameter if empty
  # if not kernel_param:
  #    kernel_param=median_heuristicsbatch(Xdata)
  #
  # #set mylambda to 1 for all view if empty
  # if not mylambda:
  #    mylambda=[1]*M
  #
  # if not num_features:
  #   n1=Yold.size(0)
  #   print('n1 is', n1)
  #   if n1>=1000:
  #     num_features=int(300)
  #   else:
  #     num_features=int(torch.floor(torch.tensor(n1/2)))
  #     print('num_features is', num_features)

  # #estimate number of components if emtpy
  # if ncomponents==0:
  #   ncomponents=chooseK(Xdata, kernel_param,eigsplot=False, TopK=20, threshold = 0.1, verbose=True)
  #   ncomponents=int(ncomponents)

  #estimate kernel parameter if empty
  if not kernel_param:
     kernel_param=median_heuristicsbatch(Xtrain)

  # if len(mylambda)==0:
  #   mylambda=list(range(M))
  #   for d in range(M):
  #       mylambda[d]=1

#   #set mylambda to 1 for all view if empty
#set mylambda to 1 for all view if empty
  if not mylambda:
     mylambda=[1]*M
  #set minRho to 0.1 for all view if empty

  if not rhoLower:
     rhoLower=[0.0000001]*M

  #set maxRho to 1 for all views if empty
  if not rhoUpper:
     rhoUpper=[0.00001]*M


  if not num_features:
    n1=Ytrain.size(0)
    print('n1 is', n1)
    if n1>=1000:
      num_features=int(300)
    else:
      num_features=int(torch.floor(torch.tensor(n1/2)))
  #estimate number of components if emtpy
  if ncomponents==0:
    ncomponents=chooseK(Xtrain, kernel_param,eigsplot=False, TopK=20, threshold = 0.1, verbose=True)
    ncomponents=int(ncomponents)

  if outcometype=='binary':
    nc=len(torch.unique(Ytrain)) #number of unique classes
    Nn=torch.zeros(nc,1)
    foldid2=list(range(nc))
    random.seed(12345)
    foldid2=[]
    for ii in range(nc):
      Nn[ii]=torch.sum(Ytrain==ii)
      foldid=torch.Tensor.repeat(torch.arange(start=0,end=nfolds),math.floor(Nn[ii]/nfolds))
      foldid2b=torch.cat([foldid, torch.arange(start=0,end=int(math.fmod(Nn[ii],nfolds)))]).tolist() #converts the tensor to a list
      foldid3b=random.sample(torch.Tensor(foldid2b).tolist(),int(Nn[ii]))
      foldid2.append(torch.Tensor(foldid3b))
      foldid3=torch.cat(foldid2,dim=-1) #concatenates the given sequence along dimension -1 (makes them into vector)


  elif outcometype=='continuous':
    foldid=torch.Tensor.repeat(torch.arange(start=0,end=nfolds),math.floor(nrow/nfolds))
    foldid2=torch.cat([foldid, torch.arange(start=0,end=int(math.fmod(nrow,nfolds)))]).tolist() #converts the tensor to a list
    random.seed(12345)
    foldid3=random.sample(foldid2,nrow)


  #cross-validation loop
  M=len(Xtrain)
  Xtrainr=list(range(M))
  Xvalidr=list(range(M))
  myKrmat=list(range(nfolds))
  foldid4=torch.Tensor(foldid3)

  myrhomin2=rhoLower
  myrhomax2=rhoUpper
  for r in range(nfolds):
    myrhomin=myrhomin2
    myrhomax=myrhomax2
    Xtrainr=[Xtrain[d][foldid4!=r,:] for d in range(M)]
    Xvalidr=[Xtrain[d][foldid4==r,:] for d in range(M)]
    Ytrainr=Ytrain[foldid4!=r,:]
    Yvalidr=Ytrain[foldid4==r,:]

    myKr=chooselassoValid2IndGroup(Xtrainr, Ytrainr,Xvalidr,Yvalidr,myseed,ncomponents,num_features,hasGroupInfo,numbercores,outcometype,
                           kernel_param,mylambda, myeta, myrhomin,myrhomax,groupsd,gridmethod,ngrid,max_iter_nsir,
                           max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)


    myresultd=torch.zeros(len(myKr[2]),M+3)
    myresultd[:,0]=int(r)*torch.ones(len(myKr[2]))
    myresultd[:,1]=torch.Tensor(myKr[1])
    myresultd[:,2]=myKr[3]
    myKrmat[r] = myresultd


  che=torch.cat(myKrmat)
  ll2=torch.unique(che[:,2])
  mymean=torch.zeros(len(ll2),M+2)
  for ll in range(len(myKr[2])):
    mymean[ll,0]=torch.mean(che[che[:,2]==ll2[ll],1])
    mymean[ll,1]=ll2[ll]
  [minError, lassoind]=torch.min(mymean[:,0:1],dim=0)
  if gridmethod=='GridSearch':
    che4=torch.Tensor(myKr[2])
  elif gridmethod=='RandomSearch':
    che4=torch.stack(myKr[2])

  minlasso=che4[lassoind,:]

  lassopenaltyd=minlasso[0]

  # #use min lasso on training data
  myalg=NSIRAlgorithmFISTASIndandGLasso(Xtrain, Ytrain,myseed, ncomponents,num_features, hasGroupInfo,
          groupsd,outcometype,kernel_param,mylambda, lassopenaltyd, myeta,max_iter_nsir,
          max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)
  return{
      'myalg':myalg,
      'Optimum.Rho':lassopenaltyd
  }
