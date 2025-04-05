import numpy as np
import sys
import config
import MAP_GaP as MG
import MAP_Rate as MR
import MAP_Hier as MH
import MAP_Shape as MS
import MAP_BGAR as MB


def column_choice(N, p):
    '''
    Selects the indices for the missing columns
    N - Number of columns in the dataset
    p - Fraction of missing data
    '''

    # Compute an even number of missing columns
    N_col = int(p*N)
    if np.mod(N_col, 2)==1:
        N_col = N_col + 1
    
    # Find indices of missing columns
    idx = []
    idx.append(N-1) # Always choose the last column
    while(len(idx) < N_col):
        i = np.random.choice(range(1,N-1), 1)[0]
        if i not in idx:
            if i-1 not in idx:
                if i+1 not in idx:
                    idx.append(i)
                    
    return np.asarray(idx, dtype = 'int64')


def prediction(path, K, p, Nmax, Nsplit, Ninit, seed):
    '''
    Runs the whole prediction experiment, with loops over splits and initializations
    path - Path to the dataset
    K - Factorization rank
    p - Fraction of missing data
    Nmax - Max. number of iterations for the MAP algorithms
    Nsplit - Number of splits
    Nitit - Number of initializations
    seed - seed
    '''

    # Load dataset
    V = np.load(path)
    F,N = V.shape

    # Results storing
    res_gap = np.zeros((2, Nsplit, Ninit))
    res_ced = np.zeros((2, Nsplit, Ninit))
    res_cd = np.zeros((2, Nsplit, Ninit))
    res_zhou = np.zeros((2, Nsplit, Ninit))
    res_bgar = np.zeros((2, Nsplit, Ninit))
    
    np.random.seed(seed)
    
    # Draw Ninit random initializations
    W_init = np.random.gamma(2, 0.5, (F,K, Ninit))
    H_init = np.random.gamma(2, 0.5, (K,N, Ninit))
    
    for i in range(0, Nsplit):
        # Draw missing columns
        idx_col = column_choice(N, p)
        M = np.ones((F,N))
        M[:, idx_col] = 0
        L = int(len(idx_col)/2)
        # Split into validation and test sets
        y_test = idx_col[:L]
        y_val = idx_col[L:]
        
        for j in range(0, Ninit):
            res_gap[:,i,j] = MG.pred(V, M, K, W_init[:,:,j], H_init[:,:,j], Nmax, y_val, y_test)
            res_ced[:,i,j] = MR.pred(V, M, K, W_init[:,:,j], H_init[:,:,j], Nmax, y_val, y_test)
            res_cd[:,i,j] = MH.pred(V, M, K, W_init[:,:,j], H_init[:,:,j], Nmax, y_val, y_test)
            res_zhou[:,i,j] = MS.pred(V, M, K, W_init[:,:,j], H_init[:,:,j], Nmax, y_val, y_test)
            res_bgar[:,i,j] = MB.pred(V, M, K, W_init[:,:,j], H_init[:,:,j], Nmax, y_val, y_test)

    # Print results
    print(r'\toprule')
    print('Model & KLE-S & KLE-F ' + r'\\')
    print(r'\midrule')
    
    print('GaP '
    + '& $' + "%.2e"%np.mean(res_gap[0,:,:]) + ' \pm ' + "%.2e"%np.std(res_gap[0,:,:]) + '$ '
    + '& $' + "%.2e"%np.mean(res_gap[1,:,:]) + ' \pm ' + "%.2e"%np.std(res_gap[1,:,:]) + '$ '
    + r'\\')
    
    print('Rate '
    + '& $' + "%.2e"%np.mean(res_ced[0,:,:]) + ' \pm ' + "%.2e"%np.std(res_ced[0,:,:]) + '$ '
    + '& $' + "%.2e"%np.mean(res_ced[1,:,:]) + ' \pm ' + "%.2e"%np.std(res_ced[1,:,:]) + '$ '
    + r'\\')
    
    print('Hier '
    + '& $' + "%.2e"%np.mean(res_cd[0,:,:]) + ' \pm ' + "%.2e"%np.std(res_cd[0,:,:]) + '$ '
    + '& $' + "%.2e"%np.mean(res_cd[1,:,:]) + ' \pm ' + "%.2e"%np.std(res_cd[1,:,:]) + '$ '
    + r'\\')
    
    print('Shape '
    + '& $' + "%.2e"%np.mean(res_zhou[0,:,:]) + ' \pm ' + "%.2e"%np.std(res_zhou[0,:,:]) + '$ '
    + '& $' + "%.2e"%np.mean(res_zhou[1,:,:]) + ' \pm ' + "%.2e"%np.std(res_zhou[1,:,:]) + '$ '
    + r'\\')
    
    print('BGAR '
    + '& $' + "%.2e"%np.mean(res_bgar[0,:,:]) + ' \pm ' + "%.2e"%np.std(res_bgar[0,:,:]) + '$ '
    + '& $' + "%.2e"%np.mean(res_bgar[1,:,:]) + ' \pm ' + "%.2e"%np.std(res_bgar[1,:,:]) + '$ '
    + r'\\')
    
    print(r'\toprule')

    return None

    
def main():
    '''
    Needs three user inputs
    path - Path to the dataset
    K - Factorization rank
    seed - seed
    '''

    path = sys.argv[1]
    K = int(sys.argv[2])
    p = config.p
    Nmax = config.Nmax
    Nsplit = config.Nsplit
    Ninit = config.Ninit
    seed = int(sys.argv[3])
    
    # Redirect print output to .txt file
    orig_stdout = sys.stdout
    f = open(path[:-4] + '_' + str(seed) + '.txt', 'w')
    sys.stdout = f
    
    # Actual experiment
    prediction(path, K, p, Nmax, Nsplit, Ninit, seed)
    
    sys.stdout = orig_stdout
    f.close()

    return None
    
    
if __name__ == "__main__":
    main()
