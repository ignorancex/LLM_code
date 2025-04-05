import numpy as np
from scipy.special import polygamma


def map_Zhou(V, M, K, a, b, W_init, H_init, Nmax):
    '''
    Algorithm for MAP estimation with missing data in temporal NMF model 'Shape'
    '''
    eps = np.spacing(1)
    F,N = V.shape
    
    W = W_init.copy()
    H = H_init.copy()

    V_hat = np.dot(W,H) + eps
    
    Nz = 5 # Newton's iterations
                
    for i in range(0,Nmax):
        # Update W
        Pp = W*np.dot((M*V)/V_hat, H.T) # P prime
        W = Pp/np.sum(Pp, axis = 0)
        V_hat = np.dot(W,H) + eps
        
        P = H*np.dot(W.T,(M*V)/V_hat)
        Q = np.dot(W.T, M)
    
        # Update h_1
        for j in range(0,Nz):
            t1 = -P[:,0] + (Q[:,0]-a*np.log(b*H[:,1] + eps))*H[:,0] + a*polygamma(0,a*H[:,0])*H[:,0]
            t2 = Q[:,0]-a*np.log(b*H[:,1] + eps) + a*(polygamma(0,a*H[:,0]) + a*H[:,0]*polygamma(1,a*H[:,0]))
            H_update = H[:,0] - t1/(t2 + eps)
            H_update[H_update < 0] = eps
            H[:,0] = H_update
        
        # Update h_n
        for n in range(1,N-1):
            for j in range(0,Nz):
                t1 = (1 - a*H[:,n-1] - P[:,n]) + (Q[:,n] + b - a*np.log(b*H[:,n+1] + eps))*H[:,n] + a*polygamma(0,a*H[:,n])*H[:,n]
                t2 = (Q[:,n] + b - a*np.log(b*H[:,n+1] + eps)) + a*(polygamma(0,a*H[:,n]) + a*H[:,n]*polygamma(1,a*H[:,n]))
                H_update = H[:,n] - t1/(t2  + eps)
                H_update[H_update < 0] = eps
                H[:,n] = H_update
        
        # Update h_N
        H[:,-1] = (P[:,-1] + a*H[:,-2] - 1)/(Q[:,-1] + b)
        H[:,-1][H[:,-1] < 0] = eps

        H = H + eps
        V_hat = np.dot(W,H) + eps

    return W, H


def pred(V, M, K, W_init, H_init, Nmax, y_val, y_test):
    '''
    Prediction experiment using MAP estimation in the temporal NMF model 'Shape'
    Returns KL errors 'S' and 'F'
    V - Data matrix
    M - Mask matrix
    K - Factorization rank
    W_init, H_init - Initialization points
    Nmax - Max. number of iterations for the MAP algorithm
    y_val - Index of columns in the validation set
    y_test - Index of columns in the test set
    '''
    F,N = V.shape
    eps = np.spacing(1)
    
    p_range = np.array([0.1, 1, 10])

    Ws = np.zeros((len(p_range), F, K))
    Hs = np.zeros((len(p_range), K, N))
    
    ds = np.zeros(len(p_range),)
            
    for i in range(0, len(p_range)):
        a = p_range[i]
        b = p_range[i]
        # Run MAP
        Ws[i,:,:], Hs[i,:,:] = map_Zhou(V, M, K, a, b, W_init, H_init, Nmax)
        V_hat = np.dot(Ws[i,:,:], Hs[i,:,:]) + eps
        # Compute KLE on validation set
        ds[i] = np.sum(V[:,y_val]*np.log((V[:,y_val]+eps)/V_hat[:,y_val]) - V[:,y_val] + V_hat[:,y_val])          

	# Find optimal hyperparameters
    i_opt = np.where(ds==ds.min())
    V_hat = np.dot(Ws[i_opt[0][0],:,:], Hs[i_opt[0][0],:,:]) + eps
    
    # Compute KLE on test set
    d_kl_s = np.sum(V[:,y_test[1:]]*np.log((V[:,y_test[1:]]+eps)/V_hat[:,y_test[1:]]) - V[:,y_test[1:]] + V_hat[:,y_test[1:]])
    d_kl_f = np.sum(V[:,-1]*np.log((V[:,-1]+eps)/V_hat[:,-1]) - V[:,-1] + V_hat[:,-1])

    return d_kl_s, d_kl_f