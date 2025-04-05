import numpy as np


def costf(V, M, W, H, B, a, b, rho):
    '''
    Compute the cost function (up to constants) for MAP estimation with missing data
    in the temporal NMF model 'BGAR'
    '''
    eps = np.spacing(1)
    g = a*(1-rho)
    e = a*rho
    Y = H[:,1:]-B[:,1:]*H[:,:-1]
    V_hat = np.dot(W,H) + eps
    a1 = -np.sum(M*(V*np.log(V_hat) - V_hat))
    a2 = np.sum(-a*np.log(b) + (1-a)*np.log(H[:,0] + eps) + b*H[:,0])
    a3 = np.sum(-g*np.log(b) + (1-g)*np.log(Y + eps) + b*Y)
    a4 = np.sum((1-e)*np.log(B[:,1:]) + (1-g)*np.log(1-B[:,1:]))
    return a1+a2+a3+a4


def map_BGAR(V, M, K, rho, a, b, W_init, H_init, Nmax):
    '''
    Algorithm for MAP estimation with missing data in temporal NMF model 'BGAR'
    '''   
    eps = np.spacing(1)
    F,N = V.shape
    
    g = a*(1-rho) # Gamma
    e = a*rho # Eta
    
    W = W_init.copy()
    H = H_init.copy()
    B = np.zeros((K,N))

    V_hat = np.dot(W,H) + eps
    
    diff = 1
    i = 0
    
    while(diff > 10**(-5) and i < Nmax):
        # Block update W
        Pp = W*np.dot((M*V)/V_hat, H.T) # P prime
        W = Pp/np.sum(Pp, axis = 0)
        V_hat = np.dot(W,H) + eps

        # Update B
        Z = H[:,1:]/(H[:,:-1] + eps)
        d3 = -b*H[:,:-1]
        d2 = 2*(1-g) + (1-e) + b*H[:,:-1]*(Z+1)
        d1 = (g+e-2)*(Z+1) - b*H[:,:-1]*Z
        d0 = (1-e)*Z
            
        for k in range(0,K):
            for n in range(1,N):
                z = np.minimum(1, Z[k,n-1])
                x = np.roots((d3[k,n-1],d2[k,n-1],d1[k,n-1],d0[k,n-1]))
                B[k,n] = x[np.where((x > 0) & (x < z))[0][0]]
                        
        # Update H
        P = H*np.dot(W.T, (M*V)/V_hat) #(K,N)
        Q = np.dot(W.T, M) #(K,N)
        
        # H_1
        d = H[:,1]/(B[:,1]+ eps)
        qt = Q[:,0] + b*(1-B[:,1])
        d2 = -qt
        d1 = -(1-a-P[:,0]) + d*qt - (1-g)
        d0 = (1-a-P[:,0])*d
        
        for k in range(0,K):
            x = np.roots((d2[k],d1[k],d0[k]))
            H[k,0] = x[np.where((x > 0) & (x < d[k]))[0][0]]

        # H_n
        for n in range(1,N-1):
            c = B[:,n]*H[:,n-1]
            d = H[:,n+1]/(B[:,n+1] + eps)
            qt = Q[:,n] + b*(1-B[:,n+1])
            d3 = -qt
            d2 = P[:,n] - 2*(1-g) + qt*(c+d)
            d1 = -P[:,n]*(c+d) + (1-g)*(c+d) - qt*c*d
            d0 = P[:,n]*c*d
            for k in range(0,K):
                x = np.roots((d3[k],d2[k],d1[k],d0[k]))
                H[k,n] = x[np.where((x > c[k]) & (x < d[k]))[0][0]]

        # H_N
        c = B[:,-1]*H[:,-2]
        qt = Q[:,-1] + b
        d2 = qt 
        d1 = -P[:,-1] -c*qt + (1-g)
        d0 = c*P[:,-1]
        
        for k in range(0,K):
            x = np.roots((d2[k],d1[k],d0[k]))
            H[k,-1] = x[np.where(x > c[k])[0][0]]
        
        V_hat = np.dot(W,H) + eps
        
        c_new = costf(V, M, W, H, B, a, b, rho)        
        if i>0:
            diff = np.abs((c_new-c_old)/c_old)
        c_old = c_new
        i = i + 1
    
    return W, H


def pred(V, M, K, W_init, H_init, Nmax, y_val, y_test):
    '''
    Prediction experiment using MAP estimation in the temporal NMF model 'BGAR'
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
    
    rho = 0.9
    a_base = 1/(1-rho) + 1
    a_range = a_base*np.array([1, 10, 100])
    b_range = np.array([0.1, 1, 10])

    Ws = np.zeros((len(a_range), len(b_range), F, K))
    Hs = np.zeros((len(a_range), len(b_range), K, N))
    
    ds = np.zeros((len(a_range), len(b_range)))
            
    for i in range(0, len(a_range)):
        a = a_range[i]
        for j in range(0, len(b_range)):
            b = b_range[j]
            # Run MAP
            Ws[i,j,:,:], Hs[i,j,:,:] = map_BGAR(V, M, K, rho, a, b, W_init, H_init, Nmax)
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