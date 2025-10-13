from __future__ import division
import pandas as pd
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import gc
import scipy.linalg as sl
import sys

def load_data(filename):
    return(pd.read_csv(filename,delimiter=',', header=0,index_col=0))


def values_matrix(dataframe):
    return(np.asarray(dataframe.as_matrix()).astype(float))


def scale(dataMatrix):
    outputData=np.copy(dataMatrix)
    means=np.mean(dataMatrix,0)
    for i in range(len(means)):
        outputData[:,i]=outputData[:,i]-means[i]
    sd=np.std(dataMatrix,0)
    for i in range(len(sd)):
        if sd[i] != 0:
            outputData[:,i]=outputData[:,i]/sd[i]
    return(outputData)


def distances_to_neighbours(scaledData,numNeighbours=10):
    [nRows,nCols]=np.shape(scaledData) 
    linksList=[]
    for ego in range(nRows):
        rowDis=np.zeros(nRows) #vector to hold the list of distances to otehr nodes
        egoVals=scaledData[ego,:]
        for target in range(nRows):
            targetVals=scaledData[target,:]
            diff=egoVals-targetVals
            diffSquared=diff**2
            euclidianDistance=np.sum(diffSquared)**(1.0/2)
            rowDis[target]=euclidianDistance

        for i in range(numNeighbours+1)[1:]:
            target=np.argsort(rowDis)[i]
            linksList.append([ego,target,rowDis[target]])
    del(rowDis)
    return(linksList)


def laplacian(linksList,numDataPoints):
    #Finds the normalized random walk graph laplacian from a linksList - the all elements of the linksList are included in the laplacian
    adjacency=ss.lil_matrix((numDataPoints,numDataPoints))
    for link in linksList:
        adjacency[link[0],link[1]]=1/link[2]
        adjacency[link[1],link[0]]=1/link[2]
        adjacency=adjacency.tocsr()
    for i,row in enumerate(adjacency):
        adjacency[i,:]=row/np.sum(row)
    return(ss.identity(numDataPoints,format='csr')-adjacency.tocsr())

def eigenvectors(laplacian,numEigenvectors):
    return(np.asarray(ssl.eigs(laplacian,k=numEigenvectors,which='SM')))


def algorithm(filename,numNeighbours=10,numEigenvectors=11):
    #performs diffusion map, taking dataframe of data and returning dataframe with numEigenvectors leading eigenvectors
    dataInput=load_data(filename)
    eigenvectorData=pd.DataFrame(index=dataInput.index.values)

    dataMatrix=values_matrix(dataInput)
    numDataPoints=len(dataInput)
    print('There are ' +str(np.shape(dataMatrix)[1]) +' varaibles for ' + str(np.shape(dataMatrix)[0]) +' data points\n') 
    del(dataInput)
    scaledData=scale(dataMatrix)
    del(dataMatrix)
    euclidianDistance=distances_to_neighbours(scaledData,numNeighbours)
    del(scaledData)
    laplacianMatrix=laplacian(euclidianDistance,numDataPoints)
    del(euclidianDistance)
    eigenvalues,eigenvectorMatrix=eigenvectors(laplacianMatrix,numEigenvectors=numEigenvectors)
    del(laplacianMatrix)
    for i,eigIndex in enumerate(np.argsort(eigenvalues)):
        
        eigenvectorData["".join(list('Eigenvector_'+str(i)))]=np.real(eigenvectorMatrix[:,eigIndex])

    eigenvectorData.to_csv(filename.split('.')[0].replace('/','-').replace('.','_')+'diffusion_map.csv')

    return()

def main(filename):
    algorithm(filename)

if __name__=='__main__':
    main(sys.argv[1])
