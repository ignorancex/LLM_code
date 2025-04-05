import numpy as np


def costf(V, M, W, H, a, b):
    '''
    Compute the cost function (up to constants) for MAP estimation with missing data
    in the temporal NMF model 'Rate'
    '''
    eps = np.spacing(1)
    V_hat = np.dot(W,H) + eps
    t1 = -np.sum(M*(V*np.log(V_hat) - V_hat))
    t2 = np.sum(-a*np.log(b) + a*np.log(H[:,:-1] + eps) + (1-a)*np.log(H[:,1:] + eps) + b*H[:,1:]/(H[:,:-1]+eps))
    return t1 + t2


def map_Rate(V, M, K, a, b, W_init, H_init, Nmax):
    '''
    Algorithm for MAP estimation with missing data in temporal NMF model 'Rate'
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
        P = H*np.dot(W.T,(M*V)/V_hat)
        Q = np.dot(W.T, M)
        
        # Update h_1
        D1 = (a-P[:,0])**2 + 4*Q[:,0]*b*H[:,1]
        H[:,0] = ((P[:,0]-a) + np.sqrt(D1))/(2*Q[:,0] + eps)
        
        # Update h_n
        D = (1-P[:,1:-1])**2 + 4*(Q[:,1:-1]+b/(H[:,:-2]+eps))*(b*H[:,2:])
        H[:,1:-1] = ((P[:,1:-1]-1) + np.sqrt(D))/(2*(Q[:,1:-1]+b/(H[:,:-2]+eps)))
        
        # Update h_N
        H[:,-1] = (P[:,-1] + a - 1)/(Q[:,-1]+b/(H[:,-2] + eps))
        H[:,-1][H[:,-1] < 0] = eps
        
        V_hat = np.dot(W,H) + eps

        # Compute cost function
        c_new = costf(V, M, W, H, a, b)
        d = np.abs((c_new-c_old)/c_old)
        c_old = c_new
        i = i + 1

    return W, H


def pred(V, M, K, W_init, H_init, Nmax, y_val, y_test):
    '''
    Prediction experiment using MAP estimation in the temporal NMF model 'Rate'
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

    Ws = np.zeros((len(p_range), F, K))
    Hs = np.zeros((len(p_range), K, N))
    
    ds = np.zeros(len(p_range),)
            
    for i in range(0, len(p_range)):
        a = p_range[i]
        b = p_range[i]
        # Run MAP
        Ws[i,:,:], Hs[i,:,:] = map_Rate(V, M, K, a, b, W_init, H_init, Nmax)
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