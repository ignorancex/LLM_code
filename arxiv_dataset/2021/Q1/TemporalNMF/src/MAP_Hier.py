import numpy as np


def costf(V, M, W, H, Z, a_z, a_h, b_z, b_h):
    '''
    Compute the cost function (up to constants) for MAP estimation with missing data
    in the temporal NMF model 'Hier'
    '''
    eps = np.spacing(1)
    V_hat = np.dot(W,H) + eps
    t1 = -np.sum(M*(V*np.log(V_hat) - V_hat))
    t2 = np.sum(-a_z*np.log(b_z*(H[:,:-1] + eps)) + (1-a_z)*np.log(Z[:,1:] + eps) + b_z*H[:,:-1]*Z[:,1:])
    t3 = np.sum(-a_h*np.log(b_h*(Z[:,1:] + eps)) + (1-a_h)*np.log(H[:,1:] + eps) + b_h*H[:,1:]*Z[:,1:])
    return t1 + t2 + t3


def map_CD(V, M, K, a_z, a_h, b_z, b_h, W_init, H_init, Nmax):
    '''
    Algorithm for MAP estimation with missing data in temporal NMF model 'Hier'
    '''
    eps = np.spacing(1)
    F,N = V.shape
    
    W = W_init.copy()
    H = H_init.copy()
    Z = np.zeros((K,N))

    V_hat = np.dot(W,H) + eps
    
    d = 1
    i = 0

    while(d > 10**(-5) and i < Nmax):
        # Update W
        Pp = W*np.dot((M*V)/V_hat, H.T)
        W = Pp/np.sum(Pp, axis = 0)
        V_hat = np.dot(W,H) + eps
        
        # Update Z
        Z[:,1:] = (a_z + a_h - 1)/(b_z*H[:,:-1] + b_h*H[:,1:] + eps)
        Z[Z<0] = eps

        # Update H
        P = H*np.dot(W.T,(M*V)/V_hat)
        Q = np.dot(W.T, M)
        
        # Update h_1
        H[:,0] = (P[:,0] + a_z)/(Q[:,0] + b_z*Z[:,1] + eps)
        H[:,0][H[:,0] < 0] = eps
        
        # Update h_n
        H[:,1:-1] = (P[:,1:-1] + a_h + a_z - 1)/(Q[:,1:-1] + b_z*Z[:,2:] + b_h*Z[:,1:-1] + eps)
        H[:,1:-1][H[:,1:-1] < 0] = eps
        
        # Update h_N
        H[:,-1] = (P[:,-1] + a_h - 1)/(Q[:,-1] + b_h*Z[:,-1] + eps)
        H[:,-1][H[:,-1] < 0] = eps

        V_hat = np.dot(W,H) + eps

        # Compute cost function
        c_new = costf(V, M, W, H, Z, a_z, a_h, b_z, b_h) 
        if i>0:
            d = np.abs((c_new-c_old)/c_old)
        c_old = c_new
        i = i + 1

    return W, H


def pred(V, M, K, W_init, H_init, Nmax, y_val, y_test):
    '''
    Prediction experiment using MAP estimation in the temporal NMF model 'Hier'
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
    
    p_range = np.array([1.5, 10, 100])

    Ws = np.zeros((len(p_range), len(p_range), F, K))
    Hs = np.zeros((len(p_range), len(p_range), K, N))
    
    ds = np.zeros((len(p_range), len(p_range)))
            
    for i in range(0, len(p_range)):
        a_z = p_range[i]
        b_z = p_range[i]
        for j in range(0, len(p_range)):
            a_h = p_range[j]
            b_h = p_range[j]
            # Run MAP
            Ws[i,j,:,:], Hs[i,j,:,:] = map_CD(V, M, K, a_z, a_h, b_z, b_h, W_init, H_init, Nmax)
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