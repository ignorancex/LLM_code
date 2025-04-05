import numpy as np
import random
from math import modf, log
from scipy.spatial.distance import cdist
from kbmom.kmedianpp import euclidean_distances, kmedianpp_init
from kbmom.utils import loglikelihood, BIC
from sklearn.metrics import davies_bouldin_score, silhouette_score

class KbMOM:
    
    def __init__(self,X,K,nbr_blocks,coef_ech = 6,max_iter = 40,outliers = None, confidence = 0.95, threshold = 0.001,quantile   = 0.5,initial_centers = None,init_type ='km++',averaging_strategy='cumul', n_layers = 1):
        '''
        # X             : numpy array = contains the data we want to cluster
        # K             : number of clusters
        # nbr_blocks    : number of blocks to create in init and loop
        # coef_ech      : NUMBER of data in each block and cluster
        # quantile      : quantile to keep for the empirical risk; by default the median
        # max_iter      : number of iterations of the algorithm
        # max_iter_init : number of iterations to run for the kmeans in the initilization procedure
        # kmeanspp      : boolean. If true then init by kmeanspp else kmedianpp
        # outliers      : number of supposed outliers
        '''
        
        '''
        # the structure of X_blocks and centers needs to to be modified.
        # they need to store the whole parameters of a model.
        # but when computing distance or predicting, just use the last
        # layer of the parameters.
        # to faciliate this computing, just added a function last_layer
        
        # Thus each item in X is a [n_layers] model, X is by number of clients array of numpy data.
        '''
        
        # given element
        self.X          = None
        self.K          = K
        self.max_iter   = max_iter
        self.n, self.p  = len(X), X[0][n_layers].shape[0] * X[0][n_layers].shape[1]
        self.quantile   = quantile
        self.coef_ech   = coef_ech
        self.B          = nbr_blocks
        self.alpha      = 1 - confidence
        self.threshold  = threshold
        self.init_type  = init_type
        self.averaging_strategy = averaging_strategy
        self.n_layers = n_layers
        
        
        # Test some given values
        if outliers is not None:
            self.outliers = outliers
            t_sup = self.bloc_size(self.n,self.outliers)
            if self.coef_ech > t_sup:
                self.coef_ech  = max((t_sup-5),1)
                self.coef_ech  = int(round(self.coef_ech))
                print('warning:: the size of blocks has been computed according to the breakdown point theory')

            B_sup = self.bloc_nb(self.n,self.outliers,b_size=self.coef_ech,alpha=self.alpha)
            if self.B < B_sup :
                self.B     = round(B_sup) + 10
                self.B     = int(self.B)
                print('warning:: the number of blocks has been computed according to the breakdown point theory')
        
        # Deal with exceptions:
        if self.coef_ech <= self.K:
            self.coef_ech = 2*self.K
        
        # internal element initialization
        self.score         = np.ones((self.n,))
        
        if isinstance(initial_centers,np.ndarray):
            self.centers = initial_centers
        else:
            self.centers = 0
            
        self.block_empirical_risk = []
        self.median_block_centers = []
        self.empirical_risk = []
        self.iter           = 1
        self.warnings       = 'None'
    
    def init_centers_function(self,X,idx_blocks):
        '''
        # Initialisation function: create nbr_blocks blocks, initialize with a kmeans++, 
        retrieve the index of the median block and its empirical risk value
        
         ``` prms ```
        . X          : numpy array of data
        . idx_blocks : list of indices contained in the B blocks
        '''
        
        # Blocks creation
        size_of_blocks = self.coef_ech
        
        block_inertia = []
        init_centers  = []
        if self.init_type=='km++':
            # instanciation of kmeans++
            x_squared = X**2
            x_squared_norms = x_squared.sum(axis=1)
        
            for idx_ in idx_blocks: 
                init_centers_ = kmedianpp_init(X[idx_,:], self.K, x_squared_norms[idx_], n_local_trials=None, square=True)
                init_centers.append(init_centers_)
                block_inertia.append(self.inertia_function(idx_,init_centers_))
        else:
            for idx_ in idx_blocks: 
                init_centers_ = self.random_init([X[i] for i in idx_])
                init_centers.append(init_centers_)
                block_inertia.append(self.inertia_function(idx_,init_centers_))
            
        median_risk = sorted(block_inertia)[round(self.quantile*len(block_inertia))]

        # Select the Q-quantile bloc
        id_median = block_inertia.index(median_risk)
        
        # init centers
        self.centers = init_centers[id_median]
        
        return(id_median,median_risk)
    
    def random_init(self,dataset):
        rnd_ =  np.random.choice(len(dataset), self.K)
        s = [dataset[i] for i in rnd_]
        return s
        
    def sampling_all_blocks_function(self):#,nbr_blocks,weighted_point,cluster_sizes):
        '''
        # Function which creates nbr_blocks blocks based on self.coef_ech and self.B
        '''
        blocks = [random.choices(np.arange(self.n),k = self.coef_ech) for i in range(self.B)]
        return(blocks)
    
    
    def inertia_function(self,idx_block,centroids = None):
        '''
        # Function which computes empirical risk per block
        
         ``` prms ```
        . X          : numpy array of data
        . idx_block  : list of indices contained in the B blocks
        . centroids  : if not None get the centers from kmeans++ initialisation
        '''
        if not isinstance(centroids,list):
            centroids = self.centers
        
#         print("The block contains:[",  idx_block ,"]")
        X_block           = [self.X[i] for i in idx_block]
        nearest_centroid  = self.fed_dist(X_block,centroids,'sqeuclidean').argmin(axis=1)
        
        if len(set(nearest_centroid)) == self.K and sum(np.bincount(nearest_centroid) > 1) == self.K :
            within_group_inertia = 0
            for k,nc in enumerate(set(nearest_centroid)):
                within_group_inertia += self.inertia_per_cluster(X_block, nearest_centroid, nc)
            
            return(within_group_inertia/len(idx_block))
        else:
            return(-1)
     
            
    def median_risk_function(self,X,blocks):
        '''
        # Function which computes the sum of all within variances and return the index of the median block
        and its empirical risk
        
        ```parameters ```       
            . blocks     : list of indices forming the blocks
            . X          : matrix of datapoints
        '''
        
        block_inertia = list(map(self.inertia_function, blocks))
            
        nb_nonvalide_blocks = sum(np.array(block_inertia) == -1)
        nb_valide_blocks    = int(self.B - nb_nonvalide_blocks)
        
        if nb_nonvalide_blocks != self.B:
            
            median_risk = sorted(block_inertia)[nb_nonvalide_blocks:][round(self.quantile*nb_valide_blocks)]
            
            # Select the Q-quantile bloc
            id_median = block_inertia.index(median_risk)
            return(id_median,median_risk)
    
        else:
            return(None,None)
        
    def medianblock_centers_function(self,X,id_median,blocks):
        '''
        #compute the barycenter of each cluster in the median block
        
         ``` prms ```
         . blocks     : list of indices forming the blocks
         . X          : matrix of datapoints
         . id_median  : index of the median block
        '''
        X_block           = [X[i] for i in blocks[id_median]]
        distances         = self.fed_dist(X_block,self.centers,'sqeuclidean')
        nearest_centroid  = distances.argmin(axis=1)
 
        print("len of nearest centroid: ", len(set(nearest_centroid)))
        centers_ = [0] * len(set(nearest_centroid))
        for k,nc in enumerate(set(nearest_centroid)):
            cl_block = []
            for i,v  in  enumerate( blocks[id_median] ) :
                if nearest_centroid[i] == nc:
                    cl_block.append(v)
            _, upd = self.E_func(self.centers[nc], cl_block)
            centers_[k] = self.M_func()
            cnt = 0
            for i, v in enumerate( blocks[id_median] ):
                if nearest_centroid[i] == nc:
                    self.X[v] = upd[cnt][1] # update is a tuple which return by E_func, the 0 is a number of samples, the 1 is model
                    cnt += 1
                             
        self.centers = centers_
        return(self)
    
    
    def weigthingscheme(self,median_block):
        '''
        Function which computes data depth
        
        ``` prms ```
        . median_block: list containing the indices of data in the median block
        
        ''' 
        for idk in median_block:
            self.score[idk] += 1
        return(self)
    
    
    def fit(self,X):
        '''
        # Main loop of the K-bmom algorithm:
        
         ``` prms ```
        . X          : matrix of datapoints 
        '''
        self.X = X
        # initialisation step
        if not isinstance(self.centers,np.ndarray):
            idx_block = self.sampling_all_blocks_function()
            id_median , median_risk_ = self.init_centers_function(X,idx_block)

            self.block_empirical_risk.append(median_risk_)
            self.medianblock_centers_function(X, id_median,idx_block)
            self.median_block_centers.append(self.centers)
            self.empirical_risk.append(sum(self.fed_dist(self.X, self.centers,'sqeuclidean').min(axis=1))/self.n)
            self.weigthingscheme(median_block=idx_block[id_median])
        
        if self.averaging_strategy == 'cumul':
            cumul_centers_ = self.centers
        
        # Main Loop - fitting process
        if (self.max_iter == 0):
            condition = False
        else:
            condition = True
       
        while condition:
            print('--- Round %d of %d: Training %d Clients ---' % (self.iter+1, self.max_iter, self.coef_ech))
            # sampling
            idx_block = self.sampling_all_blocks_function()
            
            # Compute empirical risk for all blocks and select the empirical-block
            id_median , median_risk_ = self.median_risk_function(self.X,idx_block)
            
            # If blocks are undefined, then restarting strategy
            loop_within = 0
            while (id_median == None) and loop_within < 10:
                idx_block = self.sampling_all_blocks_function()
                id_median , median_risk_ = self.init_centers_function(self.X,idx_block)
                cumul_centers_  = np.zeros((self.K,self.p))
                self.warnings = 'restart'
                loop_within += 1
            
            if id_median == None:
                self.iter = self.max_iter
                self.warnings = 'algorithm did not converge'
                condition = False
                
            else:
                # update all parameters
                self.block_empirical_risk.append(median_risk_)
                self.medianblock_centers_function(self.X,id_median,idx_block)
                self.median_block_centers.append(self.centers)
                self.empirical_risk.append(sum(self.fed_dist(self.X,self.centers,'sqeuclidean').min(axis=1))/self.n)
                self.weigthingscheme(median_block=idx_block[id_median])

#                 if self.averaging_strategy == 'cumul' and self.iter > (self.max_iter - 10):
#                     decay = self.max_iter - 10
#                     #current_centers = self.pivot(self.centers,cumul_centers_)
#                     cumul_centers_  = (self.centers / (self.iter - decay)) + (self.iter-decay - 1)/(self.iter - decay) * cumul_centers_
#                     self.centers = cumul_centers_

                # run test
                label = self.predict(self.X)
                self.T_func(label, self)
                self.iter += 1
                if self.iter>=self.max_iter:
                    condition = False
        
        return(self)
    
    
    def predict(self,X):
        '''
        Function which computes the partition based on the centroids of Median Block 
        '''
        D_nk = self.fed_dist(X,self.centers,'sqeuclidean')
        return(D_nk.argmin(axis=1))
    

    def bloc_size(self,n_sample,n_outliers):
        '''
        Function which fits the maximum size of blocks before a the breakpoint
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        '''
        return(log(2.)/log(1/(1- (n_outliers/n_sample))))


    def bloc_nb(self,n_sample,n_outliers,b_size=None,alpha=0.05):
        '''
        Function which fits the minimum nb of blocks for a given size t before a the breakpoint
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        b_size = bloc_size
        alpha : threshold confiance
        '''
        if n_outliers/n_sample >= 0.5:
            print('too much noise')
            return()
        elif b_size == None:
            t = bloc_size(n_sample,n_outliers)
            return(log(1/alpha) / (2* ((1-n_outliers/n_sample)**t - 1/2)**2))
        else:
            t = b_size
            return(log(1/alpha) / (2* ((1-n_outliers/n_sample)**t - 1/2)**2))
   
    def stopping_crit(self,risk_median):
        risk_ = risk_median[::-1][:3]
        den = (risk_[2]-risk_[1])-(risk_[1]-risk_[0])
        Ax = risk_[2] - (risk_[2]-risk_[1])**2/den
        return(Ax)
    
    def stopping_crit_GMM(self,risk_median):
        risk_ = risk_median[::-1][:3]
        Aq   = (risk_[0] - risk_[1])/(risk_[1] - risk_[2])
        
        Rinf = risk_[1] + 1/(1-Aq)*(risk_[0] - risk_[1])
        return(Rinf)
        
    def pivot(self,mu1,mu2):
        error    = cdist(mu1,mu2).argmin(axis=1)
        pivot_mu = np.zeros((self.K,self.p))
        for i,j in enumerate(error):
            pivot_mu[i,:] = mu1[j,:]
        return(pivot_mu)
    
    def set_E_func(self, func):
        self.E_func  = func
        
    def set_M_func(self, func):
        self.M_func = func
        
    def set_T_func(self, func):
        self.T_func = func
        
    def last_layer(self, X_list):
        return [x[self.n_layers] for x in X_list]
    
    def fed_dist(self, Xa, Xb, method = 'sqeuclidean'):
        xa_transformed = self.last_layer(Xa)
        xb_transformed = self.last_layer(Xb)
        
        xa = list(map(lambda x: x.flatten(), xa_transformed))
        xb = list(map(lambda x: x.flatten(), xb_transformed))
            
        return cdist(np.array(xa), np.array(xb), method)
    
    def inertia_per_cluster(self, X_block, nearest_centroids, nc):
        clster = list()
        tran_x_block = self.last_layer(X_block)
        for i, xb in enumerate(tran_x_block):
            if nearest_centroids[i] == nc:
                clster.append(xb.flatten())
                
        centers_ = np.array(clster).mean(axis = 0).reshape(1, -1)
        return cdist(np.array(clster), centers_,'sqeuclidean').sum()
    
    def npx(self):
        x_transformed = self.last_layer(self.X)
        
        x = list(map(lambda x: x.flatten(), x_transformed))
        return np.array(x)
    
    def npcenters(self):
        x_tran = self.last_layer(self.centers)
        x = list(map(lambda x: x.flatten(), x_tran))
        return np.array(x)
    
    def loglik(self):
        return loglikelihood(self.npx(), self.npcenters())

    def BIC(self):
        return BIC(self.npx(), self.npcenters())
    
    def sil_score(self):
        lbl = self.predict(self.X)
        return silhouette_score(self.npx(), lbl, metric='euclidean')

    def DB_score(self):
        lbl = self.predict(self.X)
        return davies_bouldin_score(self.npx(), lbl)
