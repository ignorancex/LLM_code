from .GPSR_Basic import *
from .OPM_recv import *

def dct_matrix(N,device,dtype=torch.float32):
    psi=torch.zeros((N,N),device=device,dtype=dtype)
    for q in range(N):
        t=np.zeros((N,1))
        t[q,0]=1
        psi[:,q]= torch.tensor(cv2.idct(t).reshape((-1,)),device=device,dtype=dtype)
    return psi