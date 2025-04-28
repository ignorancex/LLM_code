import numpy as  np
import time
## Training
from mpi4py import MPI
ROOT = 0
DONE = 999999
NOT_DONE = 1

COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
    
def produce_or_load_data(m, d):
    m_ = int(m/(SIZE-1))
    if RANK== ROOT :
        x_exact = np.random.rand(d)
        x_exact=x_exact/np.linalg.norm(x_exact)
        if x_exact.dtype != np.float32: x_exact=x_exact.astype(np.float32)
    else:
        x_exact = np.empty(d, dtype=np.float32)
        
    COMM.Bcast(x_exact, root=ROOT)
    if RANK == ROOT:
        a = None
        b = None
    else:
        a = np.random.randn(m_,d)
        a = a.astype(np.float32)
        b = (a@x_exact)**2
        b = b.astype(np.float32)
 
    return [a,b,x_exact]
  
## optimizer SGD
class solver:
    def __init__(self, w, opt_name='apam', param_type='const', period=1, alpha: float = 1e-4, amsgrad: bool = True, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-8):
        self.w = w.astype(np.float32) # here, w is the numpy array
        self.step_num = 0
        self.opt_name = opt_name
        self.alpha = alpha
        self.param_type = param_type
        
        if self.param_type == 'const':
            pass
        elif self.param_type == 'reduce' or self.param_type == 'beta_reduce' or self.param_type=='alpha_reduce':
            self.period = period
            # when period = 1, vary with respect to step
            # when period = #step in one epoch, vary with respect to epoch
        
        if self.opt_name == 'sgd':
            pass
            
        if self.opt_name == 'inertial':
            self.beta = beta1
            self.w_last = np.copy(self.w)
            self.w_temp = np.copy(self.w)
        
        if self.opt_name == 'apam':
            self.amsgrad = amsgrad
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            
            self.m = np.empty_like(self.w)
            self.v = np.empty_like(self.w)

            if self.amsgrad:
                self.v_hat = np.empty_like(self.w)

            self.reset()
              
    def reset(self):
        self.step_num = 0
        self.m = np.zeros_like(self.w, dtype=np.float32)
        self.v = np.zeros_like(self.w, dtype=np.float32)
        if self.amsgrad:
            self.v_hat = np.zeros_like(self.w, dtype=np.float32)
         
    def pack_w(self):
        w = self.w
        if w.dtype != np.float32: w=w.astype(np.float32)
        return w
         
    def step(self, g):
        self.step_num += 1
        if self.opt_name == 'sgd':
            self.w -= self.alpha*g
        
        if self.opt_name == 'inertial':
            
            if self.param_type == 'const':
                step_size = self.alpha
                moment = self.beta
            if self.param_type == 'reduce':
                epoch = int(np.floor(self.step_num/self.period))
            
                step_size = self.alpha/np.power(1+epoch,0.5)
                moment = np.min([self.beta,2/np.power(1+epoch,0.25)])
                     
            if self.param_type == 'beta_reduce':
                step_size = self.alpha

                epoch = int(np.floor(self.step_num/self.period))
                moment = np.min([self.beta,2/np.power(1+epoch,0.25)])

            if self.param_type == 'alpha_reduce':
                moment = self.beta

                epoch = int(np.floor(self.step_num/self.period))
                step_size = self.alpha/np.power(1+epoch,0.5)
         
            self.w_temp = np.copy(self.w)
            self.w += -step_size*g+moment*(self.w-self.w_last)
            self.w_last = self.w_temp
            
        if self.opt_name == 'apam':
                
            step_size = self.alpha*np.sqrt(1 - np.power(self.beta2, self.step_num))/(1 - np.power(self.beta1, self.step_num))
  
            self.m = self.m*self.beta1+(1-self.beta1)*g
            self.v = self.v*self.beta2+(1-self.beta2)*(g**2)
            if self.amsgrad:
                self.v_hat = np.maximum(self.v, self.v_hat)
                denom = np.sqrt(self.v_hat) + self.epsilon
            else:
                denom = np.sqrt(self.v) + self.epsilon
                
            self.w -= step_size*(self.m/denom)
  
def train_sync_master(num_iter_per_epoch, optimizer):
    d = len(optimizer.w)
    peers = list(range(SIZE))
    peers.remove(ROOT)
    ave_g = np.zeros(d, dtype = np.float32)
    gs = np.empty((SIZE, d), dtype = np.float32)
    num_iter = 0
     
    for i in range(num_iter_per_epoch):
        g = np.zeros(d, dtype=np.float32)
                 
        COMM.Gather(g, gs, root=ROOT)
        ave_g = gs[peers].mean(axis=0)
        optimizer.step(ave_g)
        COMM.Bcast(optimizer.pack_w(), root=ROOT)
       
def train_sync_worker(num_iter_per_epoch, a, b, batch_size, x):
    [m,d] = a.shape
    g = np.empty(d, dtype=np.float32)
    gs = None
    index = list(range(m))
    np.random.shuffle(index)
    n0 = 0
    for i in range(num_iter_per_epoch):
        a_select = a[index[n0:n0+batch_size], :]
        b_select = b[index[n0:n0+batch_size]]
        n0+= batch_size
        
        g = partial_g_fun_bs_loop(x, a_select, b_select)
        if g.dtype != np.float32: g=g.astype(np.float32)
        COMM.Gather(g, gs, root=ROOT)
        COMM.Bcast(x, root=ROOT)
    
    return x

def train_async_master(num_iter_per_epoch, optimizer):

    d = len(optimizer.w)
    peers = list(range(SIZE))
    peers.remove(ROOT)
    N_peers = len(peers)
      
    if RANK == ROOT:
        gg = np.empty((N_peers,d), dtype=np.float32)

    requests = [MPI.REQUEST_NULL for i in peers]
    for i in range(N_peers):
        requests[i] = COMM.Irecv(gg[i], source=peers[i])

    n_master_receive_each_epoch = 0
    num_active_workers = N_peers

    while  num_active_workers > 0:
        idx_of_received_list = MPI.Request.Waitsome(requests)
        for i in idx_of_received_list:
            optimizer.step(gg[i])
            n_master_receive_each_epoch += 1

            if n_master_receive_each_epoch < num_iter_per_epoch:
                COMM.Send(optimizer.pack_w(), dest=peers[i], tag=NOT_DONE)
                requests[i] = COMM.Irecv(gg[i], source=peers[i])
            else:
                COMM.Send(optimizer.pack_w(), dest=peers[i], tag=DONE)
                num_active_workers -= 1

def train_async_worker(num_iter_per_epoch, a, b, batch_size, x):

    [m,d] = a.shape

    g = np.empty(d, dtype=np.float32)
    info = MPI.Status()
    info.tag = NOT_DONE
     
    index = list(range(m))
    np.random.shuffle(index)
    n0 = 0
    
    while info.tag == NOT_DONE:
        if n0+batch_size > m:
            np.random.shuffle(index)
            n0 = 0
            
        a_select = a[index[n0:n0+batch_size], :]
        b_select = b[index[n0:n0+batch_size]]
        n0 += batch_size
        
        g = partial_g_fun_bs_loop(x,a_select,b_select)
        if g.dtype != np.float32: g = g.astype(np.float32)
        COMM.Send(g, dest=ROOT)
        COMM.Recv(x, source=ROOT, tag=MPI.ANY_TAG, status=info)
        
    return x
 
def partial_g_fun_bs_loop(x, a, b):
    [m, d] = a.shape
    grad_sum = np.zeros(d, dtype=np.float64)
    
    for m_ in range(m):
        ax_ = a[m_,:]@x
        sign_ = np.sign(ax_**2-b[m_])
        if sign_ == 0:
            ax_ *= 2*(2*np.random.rand()-1)
        else:
            ax_ *= 2*sign_
          
        grad_sum += ax_*a[m_,:]
    
    return grad_sum/m
 
## for a batch size of data
#def partial_g_fun_bs(x, a, b):
#    ax = a@x
#    sign_ax2_b = np.sign(ax**2-b)
#    tag = (sign_ax2_b == 0)
#    ax[tag]  *= 2*(2*np.random.rand(np.sum(tag))-1)
#    ax[~tag] *= 2*sign_ax2_b[~tag]
#
#    return np.mean((a.T*ax).T,0)
#
## only for one data
#def partial_g_fun(x, a, b):
#    ax = a@x
#    if ax**2==b:
#        return 2*ax*(2*np.random.rand()-1)*a
#    else:
#        return 2*ax*np.sign(ax**2-b)*a
        
        
#### hyper-parameters
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--m', type=int, default=10000)
parser.add_argument('--d', type=int, default=4000)

## optimizer
parser.add_argument('--alpha', type=float, default=0.000002)
parser.add_argument('--amsgrad', type=bool, default=True)
parser.add_argument('--beta1', type=float, default=0.8)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--eps', type=float, default=1e-8)

## loader
parser.add_argument('--train_batch_size', type=int, default=1)

## epoch and epochs
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--log_per_epoch', type=int, default=1)
 
##  method
# communication
parser.add_argument('--communication', type=str, default='sync', choices=['sync', 'async'])
# optimizer
parser.add_argument('--opt_name', type=str, default='inertial', choices=['sgd', 'inertial', 'apam'])
 
parser.add_argument('--param_type', type=str, default='const', choices=['const', 'reduce', 'beta_reduce', 'alpha_reduce'])
parser.add_argument('--period_type', type=str, default='epoch', choices=['epoch', 'step'])

# save the results
parser.add_argument('--save_dir', type=str, default='./results/')

def main():
    args = parser.parse_args()
    # parameters for optimizer
    
    param_type = args.param_type
    period_type = args.period_type
    
    opt_name = args.opt_name
    alpha = args.alpha
    amsgrad = args.amsgrad
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.eps
       
    # epochs
    num_epochs = args.epochs
    epoch = args.epoch
    
    # set randomness seed
    seed = RANK*100+20210406
    np.random.seed(seed)
      
    m = args.m
    d = args.d
    [a, b, x_exact] = produce_or_load_data(m, d)
    
    debug=False
    if debug:
        print('RANK=', RANK, 'has b:', b.size, flush=True)
        
    if not (RANK == ROOT):
        f_fun = lambda x: np.mean(np.abs((a@x)**2-b))
        f_exact = f_fun(x_exact) # 0.0
        test_fun = lambda x: np.abs(f_fun(x) - f_exact)
 
    COMM.Barrier()
          
    train_batch_size = args.train_batch_size
    Nworkers = SIZE-1
    if args.communication == 'sync':
        my_train_batch_size = train_batch_size//Nworkers
        if  RANK < train_batch_size%Nworkers and RANK != ROOT:
            my_train_batch_size = my_train_batch_size+1
        
    if args.communication == 'async':
        my_train_batch_size = train_batch_size
          
    num_iter_per_epoch = np.int(np.ceil(m/train_batch_size))
 
     # initial value x.
    if RANK == ROOT:
        x = np.random.rand(d)
        x = x/np.linalg.norm(x)
    else:
        x = np.empty(d)
        
    x = x.astype(np.float32)
    COMM.Bcast(x, root=ROOT)
    
    if period_type == 'step':
        period = 1
    elif period_type == 'epoch':
        period = num_iter_per_epoch
 
    #print('RANK=', RANK, ',PERIOD_TYPE',period_type, flush=True)
    optimizer = solver(x, opt_name=opt_name, alpha=alpha, param_type=param_type, period=period, amsgrad=amsgrad, beta1=beta1, beta2=beta2, epsilon=epsilon)
   
    if RANK == ROOT:
        filename = args.save_dir + 'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_'+opt_name+'_'+args.communication+'_epochs'+str(num_epochs) +'_SIZE_'+str(SIZE)+'_bacthsize'+str(train_batch_size)+'_param_type_'+param_type
        
        if param_type == 'reduce' or param_type == 'beta_reduce' or param_type == 'alpha_reduce':
            filename = filename + '_period_type_' + period_type
        
        filename = filename+'_alpha'+str(alpha)+'_beta'+str(beta1)+'.txt'
        
        f = open(filename,"a")
        #f.write(filename)
        f.write('epoch \t norm_x_xexact \t function_gap \t time_since_begin(without testing)\n')
        
        print(filename+ ' is computing ..... ', flush=True)
        print('epoch  \t norm_x_xexact \t function_gap \t time_since_begin(without testing)', flush=True)
    
    
    if RANK == ROOT:
        total_test_time = 0.
        time_start = time.time()
        test_time0 = time.time()
        
        peers = list(range(SIZE))
        peers.remove(ROOT)
        
        function_gap = np.mean(0.)
        function_gap = function_gap.astype(np.float32)
        function_gaps = np.empty(SIZE, dtype=np.float32)
        COMM.Gather(function_gap, function_gaps, root=ROOT)
        
        function_gap_avg = function_gaps[peers].mean(axis=0)
 
        total_test_time += time.time()-test_time0
        print('{}\t{}\t{}\t{}'.format(epoch, np.linalg.norm(x-x_exact), function_gap_avg, time.time()-time_start-total_test_time), flush=True)
        f.write('{}\t{}\t{}\t{}\n'.format(epoch,np.linalg.norm(x-x_exact),function_gap_avg, time.time()-time_start-total_test_time))
        
    else:
        function_gap = test_fun(x)
        function_gap = function_gap.astype(np.float32)
        function_gaps = None
        COMM.Gather(function_gap, function_gaps, root=ROOT)
   
          
    while epoch<num_epochs:
        
        if args.communication == 'sync':
            if RANK == ROOT:
                train_sync_master(num_iter_per_epoch, optimizer)
            else:
                x = train_sync_worker(num_iter_per_epoch, a, b, my_train_batch_size, x)
                
        if args.communication == 'async':
            if RANK == ROOT:
                train_async_master(num_iter_per_epoch, optimizer)
            else:
                x = train_async_worker(num_iter_per_epoch, a, b, my_train_batch_size, x)
                
             
        epoch += 1
        if RANK == ROOT: x = optimizer.pack_w()
        COMM.Bcast(x, root=ROOT)
        
        if epoch%args.log_per_epoch == 0:
            if RANK == ROOT:
                test_time0 = time.time()
                COMM.Gather(function_gap, function_gaps, root=ROOT)
                function_gap_avg = function_gaps[peers].mean(axis=0)
                total_test_time += time.time() - test_time0
                             
                print('{}\t{}\t{}\t{}'.format(epoch,np.linalg.norm(x-x_exact), function_gap_avg, time.time()-time_start-total_test_time), flush=True)
                f.write('{}\t{}\t{}\t{}\n'.format(epoch,np.linalg.norm(x-x_exact), function_gap_avg, time.time()-time_start-total_test_time))
            else:
                function_gap = test_fun(x)
                function_gap = function_gap.astype(np.float32)
                COMM.Gather(function_gap,function_gaps,root=ROOT)

if __name__ == '__main__':
    main()
