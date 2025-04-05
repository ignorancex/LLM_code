import numpy as np


def costf(V, M, W, H, a, b):
    '''
    Compute the cost function (up to constants) for MAP estimation with missing data
    in the GaP model
    '''
    eps = np.spacing(1)
    V_hat = np.dot(W,H) + eps
    return -np.sum(M*(V*np.log(V_hat) - V_hat)) + np.sum(-a*np.log(b) +(1-a)*np.log(H+eps) + b*H)


def map_GaP(V, M, K, a, b, W_init, H_init, Nmax):
    '''
    Algorithm for MAP estimation with missing data in the GaP model
    '''
    eps = np.spacing(1)
    F,N = V.shape
        
    W = W_init.copy()
    H = H_init.copy()
    
    V_hat = np.dot(W,H) + eps
    
    c_old = costf(V, M, W, H, a, b)
    d = 1
    i = 0
    
    while(d > 10**(-5) and i < Nmax):
        # Update W
        Pp = W*np.dot((M*V)/V_hat, H.T) # P prime
        W = Pp/np.sum(Pp, axis = 0)
        V_hat = np.dot(W,H) + eps
                
        # Update H
        H = (H*np.dot(W.T, (M*V)/V_hat) + a - 1)/(b + np.dot(W.T, M) + eps)
        H[H < 0] = eps
        V_hat = np.dot(W,H) + eps
        
        # Compute cost function
        c_new = costf(V, M, W, H, a, b)
        d = np.abs((c_new-c_old)/c_old)
        c_old = c_new
        
        i = i + 1
        
    return W, H


def pred(V, M, K, W_init, H_init, Nmax, y_val, y_test):
    '''
    Prediction experiment using MAP estimation in the GaP model
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
    
    a_range = np.array([0.1, 1, 10])
    b_range = np.array([0.1, 1, 10])

    Ws = np.zeros((len(a_range), len(b_range), F, K))
    Hs = np.zeros((len(a_range), len(b_range), K, N))
    
    ds = np.zeros((len(a_range), len(b_range)))
            
    for i in range(0, len(a_range)):
        a = a_range[i]
        for j in range(0, len(b_range)):
            b = b_range[j]
            # Run MAP
            Ws[i,j,:,:], Hs[i,j,:,:] = map_GaP(V, M, K, a, b, W_init, H_init, Nmax)
            # Estimate H for missing columns
            Hs[i,j,:,y_val] = 0.5*(Hs[i,j,:,y_val-1] + Hs[i,j,:,y_val+1])
            Hs[i,j,:,y_test[1:]] = 0.5*(Hs[i,j,:,y_test[1:]-1] + Hs[i,j,:,y_test[1:]+1])
            Hs[i,j,:,-1] = Hs[i,j,:,-2]
            V_hat = np.dot(Ws[i,j,:,:], Hs[i,j,:,:]) + eps
            # Compute KLE on validation set
            ds[i,j] = np.sum(V[:,y_val]*np.log((V[:,y_val]+eps)/V_hat[:,y_val]) - V[:,y_val] + V_hat[:,y_val])

	# Find optimal hyperparameters
    i_opt = np.where(ds==ds.min())
    V_hat = np.dot(Ws[i_opt[0][0],i_opt[1][0],:,:], Hs[i_opt[0][0],i_opt[1][0],:,:]) + eps
    
    # Compute KLE on test set
    d_kl_s = np.sum(V[:,y_test[1:]]*np.log((V[:,y_test[1:]]+eps)/V_hat[:,y_test[1:]]) - V[:,y_test[1:]] + V_hat[:,y_test[1:]])
    d_kl_f = np.sum(V[:,-1]*np.log((V[:,-1]+eps)/V_hat[:,-1]) - V[:,-1] + V_hat[:,-1])

    return d_kl_s, d_kl_f