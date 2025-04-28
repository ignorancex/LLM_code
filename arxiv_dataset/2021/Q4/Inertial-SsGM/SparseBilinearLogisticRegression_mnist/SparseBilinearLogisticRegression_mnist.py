import numpy as  np
import torch

import time
 
### define the newtwork architecture
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init

import sys

class Bilinear_2dimage(nn.Module):
    r"""Applies a bilinear transformation to the incoming 2d images:
    :math:`y_i = tr(U_i^T X V_i) + b_i` for i = 0,1,2,...,#class

    Args:
        in1: #raws of input X
        in2: #columns of input X
        r:   U_i.shape=(in1,r), V_i.shape=(in2,r). default r=1
        out: #class
        
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: (N,1,in1,in2), N=#samples,1=#channel
        - Output: (N,out)

    Attributes:
        weight_U: the learnable weights U of the module of shape
            :math:`(\text{out}, \text{in1}, \text{r})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1}}`
        weight_V: the learnable weights V of the module of shape
            :math:`(\text{out}, \text{in2}, \text{r})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in2}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1}}`

    """
    __constants__ = ['in1', 'in2', 'r', 'out']
    in1: int
    in2: int
    r: int
    out: int
    weight_U: Tensor
    weight_V: Tensor
        

    def __init__(self, in1: int, in2: int, out: int, r: int = 1, bias: bool = True) -> None:
        super(Bilinear_2dimage, self).__init__()
        self.in1 = in1
        self.in2 = in2
        self.r = r
        self.out = out
        
        self.weight_U = Parameter(torch.Tensor(out, r, in1))
        self.weight_V = Parameter(torch.Tensor(out, in2, r))

        if bias:
            self.bias = Parameter(torch.Tensor(out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound1 = 1 / np.sqrt(self.in1)
        bound2 = 1 / np.sqrt(self.in2)
        torch.nn.init.uniform_(self.weight_U, -bound1, bound1)
        torch.nn.init.uniform_(self.weight_V, -bound2, bound2)
        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1]!=1:
            sys.exit('The Bi-linear model can only deal with 2 dimensional images, the images channel should be one ')
 
        output=torch.diagonal(self.weight_U@images@self.weight_V,dim1=2,dim2=3).sum(2)+self.bias
        
        return F.log_softmax(output, dim=1)

    def extra_repr(self) -> str:
        return 'in1 ={}, in2 ={}, out ={}, r={}, bias={}'.format(
            self.in1 , self.in2 , self.out, self.r, self.bias is not None
        )

def prox_norm_1(w, lam):
    temp = (np.abs(w)-lam)
    temp[temp<0]=0
    return np.sign(w)*temp

### define the optimizer
from torch.optim.optimizer import Optimizer
from typing import  Any, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]] # Union[X, Y] means either X or Y.
OptFloat = Optional[float]

## optimizer SGD
from typing import Iterable
from torch import Tensor
Params = Iterable[Tensor]
class solver_sparse:
    def __init__(self, params:Params, device, lam=0, opt_name='apam', param_type='const', period=1, alpha: float = 1e-4, amsgrad: bool = True, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-8):
        self.params = list(params)
        self.device = device
        
        self.lam = lam # weights for the 1-norm regularizer
        
        self.num_set = len(self.params)
        self.set_size = []
        self.set_shape = []
        for param in self.params:
            self.set_size.append(param.data.numel())
            self.set_shape.append(param.data.cpu().numpy().shape)
        self.num_param = sum(self.set_size)
        
        
        self.step_num = 0
        
        self.opt_name = opt_name
        self.alpha = alpha
        
        self.param_type = param_type
        
        if self.param_type == 'const':
            pass
        elif self.param_type == 'reduce' or self.param_type == 'beta_reduce' or self.param_type=='alpha_reduce':
            self.period = period
        
        if self.opt_name == 'sgd':
            pass
            
        if self.opt_name == 'inertial':
            self.beta = beta1
            #self.w_last = np.empty(self.num_param)
            self.w_last = np.concatenate([param.data.cpu().numpy().flatten() for param in self.params ])
            self.w_temp = np.concatenate([param.data.cpu().numpy().flatten() for param in self.params ])
        
        if self.opt_name == 'apam':
            self.amsgrad = amsgrad
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            
            self.m = np.empty(self.num_param)
            self.v = np.empty(self.num_param)

            if self.amsgrad:
                self.v_hat = np.empty(self.num_param)

            self.reset()
              
    def reset(self):
        self.step_num = 0
        self.m = np.zeros(self.num_param, dtype=np.float32)
        self.v = np.zeros(self.num_param, dtype=np.float32)
        if self.amsgrad:
            self.v_hat = np.zeros(self.num_param, dtype=np.float32)
            
            
    def unpack(self, w): # unpack from float array (w) to tensor (in the model)
        offset = 0
        for idx,param in enumerate(self.params):
            param.data.copy_(torch.tensor(w[offset:offset+self.set_size[idx]].reshape(self.set_shape[idx])).to(self.device))
            offset += self.set_size[idx]
          
    def pack_w(self): # pack from tensor (parameters in the model) to float array (change w)
        w = np.concatenate([param.data.cpu().numpy().flatten() for param in self.params ])
        if w.dtype != np.float32: w=w.astype(np.float32)
        return w
 
    
    def pack_g(self):  # pack from tensor (gradient in the model) to float array (change g)
        g = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.params ])
        if g.dtype != np.float32: g=g.astype(np.float32)
        return g
    
    def zero_grad(self):
        for idx,param in enumerate(self.params):
            if param.grad is None:
                continue
            param.grad.data.copy_(torch.tensor(np.zeros(self.set_shape[idx])))
    
    def step(self, w, g):
        self.step_num += 1
        if self.opt_name == 'sgd':
            step_size = self.alpha
            w -= step_size*g
        
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


            self.w_temp = np.copy(w)
            w += -step_size*g+moment*(w-self.w_last) # let change the value on the input w's address by += 
            self.w_last = self.w_temp
 
        if self.opt_name == 'apam':
                
            step_size = self.alpha*np.sqrt(1 - np.power(self.beta2, self.step_num)) / (1 - np.power(self.beta1, self.step_num))
  
            self.m = self.m*self.beta1+(1-self.beta1)*g
            self.v = self.v*self.beta2+(1-self.beta2)*(g**2)
            if self.amsgrad:
                self.v_hat = np.maximum(self.v, self.v_hat)
                denom = np.sqrt(self.v_hat) + self.epsilon
            else:
                denom = np.sqrt(self.v) + self.epsilon
                
            w -= step_size*(self.m/denom)
            
        return prox_norm_1(w,self.lam*step_size)

## Training
from mpi4py import MPI
ROOT = 0
DONE = 999999
NOT_DONE = 1

COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
  
def train_sync_master(num_iter_per_epoch, optimizer):

    peers = list(range(SIZE))
    peers.remove(ROOT)

    w = np.zeros(optimizer.num_param, dtype=np.float32)
    ave_g = np.zeros(optimizer.num_param, dtype=np.float32)
    gs = np.empty((SIZE, optimizer.num_param), dtype=np.float32)
        
    debug = False
    
    num_iter=0
    w = optimizer.pack_w()
    
    for i in range(num_iter_per_epoch):
        g = np.zeros(optimizer.num_param, dtype=np.float32)
                
        if debug: print('0, before gather, g=', ave_g[0:5])
        if debug: print('0, before gather, gs=',gs[:,0:5])
        COMM.Gather(g, gs, root=0)
        if debug: print('0, after gather, g=', ave_g[0:5])
        if debug: print('0, after gather, gs=',gs[:,0:5])
        
        ave_g = gs[peers].mean(axis=0)# np.sum(gs,axis=0)/(SIZE-1)
        if debug: print('0,avg_g=', ave_g[0:5])
        
        if debug: print('0,befor step w=', w[0:5])
        w = optimizer.step(w,ave_g)
        if debug: print('opt_name:', optimizer.opt_name, 'alpha:', optimizer.alpha)
        if debug: print('0,after step w=', w[0:5])
        COMM.Bcast(w, root=ROOT)
    
    optimizer.unpack(w) # check if it still work without this step
 
def train_sync_worker(model, device, num_iter_per_epoch, train_loader, optimizer):
    
    w = np.empty(optimizer.num_param, dtype=np.float32)
    g = np.empty(optimizer.num_param, dtype=np.float32)
    gs = None
    
    debug = False
    model.train()
    
    dataiter = train_loader.__iter__()
    
    for i in range(num_iter_per_epoch):
        try:
            data, target = dataiter.next()
        except StopIteration:
            del dataiter
            dataiter = train_loader.__iter__()
            data, target = dataiter.next()
            
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        g = optimizer.pack_g()
        
        if debug and RANK==1: print('rank=', RANK, ',has loss=', loss.item(), flush=True)
        if debug: print('Rank:', RANK, ' before Gather has g', g[0:5])
        
        COMM.Gather(g, gs, root=0)
        
        if debug:
            print('Rank:', RANK, ' after Gather has g', g[0:5])
            w = optimizer.pack_w()
            print('Rank:', RANK, ' before bcast has w', w[0:5])
    
        COMM.Bcast(w, root=ROOT)
        if debug: print('Rank:', RANK, ' after bcast has w', w[0:5])
        optimizer.unpack(w)
 
def train_async_master(num_iter_per_epoch, optimizer):

    peers = list(range(SIZE))
    peers.remove(ROOT)
    N_peers = len(peers)
     
    w = np.empty(optimizer.num_param, dtype=np.float32)
    g = np.empty(optimizer.num_param, dtype=np.float32)
    if RANK == ROOT:
        gg = np.empty((N_peers, optimizer.num_param), dtype=np.float32)
        
    requests = [MPI.REQUEST_NULL for i in peers]
    for i in range(N_peers):
        requests[i] = COMM.Irecv(gg[i], source=peers[i])
        
            
    n_master_receive_each_epoch = 0
    w = optimizer.pack_w()
    num_active_workers = N_peers
    while  num_active_workers > 0:
        idx_of_received_list = MPI.Request.Waitsome(requests)
           
        for i in idx_of_received_list:
            w = optimizer.step(w,gg[i]) # in prox_1_norm, w is point to a new address
            
            n_master_receive_each_epoch += 1
          
            if n_master_receive_each_epoch < num_iter_per_epoch:
                COMM.Send(w, dest=peers[i], tag=NOT_DONE)
                requests[i] = COMM.Irecv(gg[i], source=peers[i])
            else:
                COMM.Send(w, dest=peers[i], tag=DONE)
                num_active_workers -= 1

    optimizer.unpack(w)

def train_async_worker(model, device, train_loader, optimizer):
    
    w = np.empty(optimizer.num_param, dtype=np.float32)
    w = optimizer.pack_w()
    g = np.empty(optimizer.num_param, dtype=np.float32)
    info = MPI.Status()
    info.tag = NOT_DONE
    
    dataiter = train_loader.__iter__()
    debug = False
    if debug: num_send_recv=0
    model.train()
    while info.tag == NOT_DONE:
        try:
            data, target = dataiter.next()
        except StopIteration:
            del dataiter
            dataiter = train_loader.__iter__()
            data, target = dataiter.next()
            
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        if debug:
            num_send_recv += 1
            
        g = optimizer.pack_g()
        COMM.Send(g, dest=ROOT)
         
        COMM.Recv(w, source=ROOT, tag=MPI.ANY_TAG, status=info)
        optimizer.unpack(w)
    
    if debug:
        print('In this epoch, rank=', RANK, 'has send/recv',num_send_recv, 'times, and loss= ', loss.item(), flush=True)
     
 
def test_sparse(model, device, test_loader, lam, optimizer):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
         
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            
            test_loss += loss.item()  #note loss is sum in the above nll_loss ,sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_loss += lam*np.linalg.norm(optimizer.pack_w(),1)
        test_acc = 100.*correct/len(test_loader.dataset)
        
    return  [test_loss, test_acc]

#### hyper-parameters
import argparse
parser = argparse.ArgumentParser()
## dataset
parser.add_argument('--data_dir', type=str, default='../data')

## optimizer
parser.add_argument('--lam', type=float, default=0.01) # the weight for the sparse regularizer

parser.add_argument('--alpha', type=float, default=0.0001)
parser.add_argument('--amsgrad', type=bool, default=True)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)

## loader
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--pred_train_batch_size', type=int, default=1000)

## epoch and epochs
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--log_per_epoch', type=int, default=5)
 
parser.add_argument('--r', type=int, default=1)
 
##  method
# communication
parser.add_argument('--communication', type=str, default='async', choices=['sync', 'async'])
# optimizer
parser.add_argument('--opt_name', type=str, default='apam', choices=['sgd', 'inertial', 'apam'])

parser.add_argument('--param_type', type=str, default='const', choices=['const', 'reduce', 'beta_reduce', 'alpha_reduce'])
parser.add_argument('--period_type', type=str, default='epoch', choices=['epoch', 'step'])
 
# save the results
parser.add_argument('--save_dir', type=str, default='./results/')

def main():
    args = parser.parse_args()
    # parameters for optimizer
    
    param_type = args.param_type
    period_type = args.period_type
    
    lam = args.lam
     
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
    seed = RANK*100+20220607
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    use_cuda= False
    # device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    
    # load datasets
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                             transform= transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))]))

    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                             transform= transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))]))
    
    num_class = 10
    [_,in1,in2] = test_dataset[0][0].shape
    model = Bilinear_2dimage(in1, in2, num_class,r=args.r).to(device)
  
    # parameters for loader
    train_batch_size = args.train_batch_size
    pred_train_batch_size = args.pred_train_batch_size
    test_batch_size = args.test_batch_size
     
    Nworkers = SIZE-1
    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and use_cuda) else {}
    if args.communication == 'sync':
        my_train_batch_size = train_batch_size//Nworkers
        if  RANK < train_batch_size%Nworkers and RANK != ROOT:
            my_train_batch_size = my_train_batch_size+1
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=my_train_batch_size, shuffle=True, **kwargs)
    if args.communication == 'async':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        
    pred_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pred_train_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    
    num_iter_per_epoch = np.int(np.ceil(len(train_dataset)/train_batch_size))
 
    if period_type == 'step':
        period = 1
    elif period_type == 'epoch':
        period = num_iter_per_epoch
 
    optimizer = solver_sparse(model.parameters(), device=device, lam=lam, opt_name=opt_name, alpha=alpha, param_type=param_type, period = period, amsgrad=amsgrad, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    w = np.empty(optimizer.num_param, dtype=np.float32)
    if RANK == ROOT:
        filename = args.save_dir + 'results_cpu_mnist_mpi_Bilinear_r_'+str(args.r)+'_lam'+str(lam)+'_'+opt_name+'_'+args.communication+'_epochs'+str(num_epochs) +'_SIZE_'+str(SIZE)+'_bacthsize'+str(train_batch_size) +'_param_type_'+param_type
        
        if param_type == 'reduce' or param_type == 'beta_reduce':
            filename = filename + '_period_type_' + period_type
        
        filename = filename +'_alpha'+str(alpha)+'_beta'+str(beta1)+'.txt'
        
        f = open(filename,"a")
        f.write('epoch\t test_loss\t test_acc\t train_loss\t train_acc \t time_since_begin(without testing)\n')
        
        print(filename+ ' is computing ..... ', flush=True)
        print('epoch\t test_loss\t test_acc\t train_loss\t train_acc \t time_since_begin(without testing)', flush=True)
        
        w = optimizer.pack_w()
    COMM.Bcast(w, root=ROOT)
    optimizer.unpack(w)
      
    total_test_time = 0.
    time_start = time.time()
            
    if RANK == ROOT:
        test_time0 = time.time()
        [test_loss, test_acc] = test_sparse(model, device, test_loader, lam, optimizer)
        [train_loss, train_acc] = test_sparse(model, device, pred_train_loader, lam, optimizer)
        total_test_time += time.time()-test_time0
        print('{}\t {}\t {}\t {}\t {}\t{}'.format(epoch, test_loss, test_acc, train_loss, train_acc,time.time()-time_start-total_test_time), flush=True)
        f.write('{}\t {}\t {}\t {}\t {}\t{}\n'.format(epoch, test_loss, test_acc, train_loss, train_acc, time.time()-time_start-total_test_time))
   
    while epoch<num_epochs:

        if args.communication == 'sync':
            if RANK == ROOT:
                train_sync_master(num_iter_per_epoch, optimizer)
            else:
                train_sync_worker(model, device, num_iter_per_epoch, train_loader, optimizer)
                
        if args.communication == 'async':
            if RANK == ROOT:
                train_async_master(num_iter_per_epoch, optimizer)
            else:
                train_async_worker(model, device, train_loader, optimizer)

        epoch += 1
        if RANK == ROOT and (epoch%args.log_per_epoch == 0 or epoch <= 20):
            test_time0 = time.time()
             
            [test_loss, test_acc] = test_sparse(model, device, test_loader, lam, optimizer)
            [train_loss, train_acc] = test_sparse(model, device, pred_train_loader, lam, optimizer)
             
            total_test_time += time.time()-test_time0
                         
            print('{}\t {}\t {}\t {}\t {}\t{}'.format(epoch, test_loss, test_acc, train_loss, train_acc,time.time()-time_start-total_test_time), flush=True)
            f.write('{}\t {}\t {}\t {}\t {}\t{}\n'.format(epoch, test_loss, test_acc, train_loss, train_acc, time.time()-time_start-total_test_time))
             
        if RANK == ROOT:  w = optimizer.pack_w()
        COMM.Bcast(w, root=ROOT)
        optimizer.unpack(w)
             
    time_end = time.time()
    if RANK == ROOT:
        print('process {}: training time {} seconds (with {} workers)\n'.format(RANK, time_end-time_start-total_test_time, Nworkers))
        print('process {}: total time {} seconds (with {} workers)\n\n'.format(RANK, time_end-time_start, Nworkers))
        f.write('process {}: Training time {} seconds (with {} workers)\n'.format(RANK, time_end-time_start-total_test_time, Nworkers))
        f.write('process {}: total time {} seconds (with {} workers)\n\n'.format(RANK, time_end-time_start, Nworkers))
        f.close()
        
if __name__ == '__main__':
    main()
