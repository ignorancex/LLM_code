import torch
import time

"""
Input type: torch.tensor
"""

def calc_iterative_algo_k(A, PM_image, x_u, _L=None, float64=True):
    if float64:
        PM_image = PM_image.to(torch.float64)
        x_u = x_u.to(torch.float64)
        A = A.to(torch.float64)
        
    L = torch.inverse(A @ A.T) if _L is None else _L
    y_0 = A @ PM_image
    return (y_0.T @ L @ A @ (PM_image - x_u)) / (y_0.T @ L @ y_0)


def speckle_measure(A, x, nosie=False, nosie_sigma=0.5):
    y = A @ x
    if nosie:
        y += nosie_sigma * torch.randn_like(y)
    return y
    

def mismatch_equation(y_0, y, A, float64=False, _L=None):
    if float64:
        y_0 = y_0.to(torch.float64)
        y = y.to(torch.float64)
        A = A.to(torch.float64)
    L = torch.inverse(A @ A.T) if _L is None else _L
    ND = 1 / (y_0.T @ L @ y_0)
    return ND * (y @ y_0.T @ L @ A)


def iterative_algo_1(y_u, params, verbose=True):
    A = params["A"]
    PM_image = params["PM_image"]
    epoch = params["epoch"]
    speckle_measure = params["speckle_measure"]
    init_A_recv = params["A_recv"]
    init_e_y = params["e_y"]
    calc_error = params["calc_error"]
    
    y_0 = A @ PM_image
    
    A_recv = torch.zeros_like(A) if init_A_recv is None else init_A_recv
    e_y = y_u if init_e_y is None else init_e_y
    
    errors = []    
    for i in range(epoch):
        A_recv_e_y = mismatch_equation(y_0, e_y, A)
        A_recv += A_recv_e_y
        e_y = y_u - speckle_measure(A_recv)
        e = abs((y_u - calc_error(A_recv)).cpu().numpy()).mean()
        if verbose:
            print("========error-{}==========".format(i))
            print(e)
        errors.append(e)
        
    return A_recv, errors



def iterative_algo_pm(A, **params):
    PM_image = params["PM_image"]
    epoch = params["epoch"]
    verbose =  params["verbose"]
    isfloat64 = params["isfloat64"]
    
    y_0 = A @ PM_image

    if isfloat64:
        A_recv = mismatch_equation(y_0, y_0, A)
        return (y_0, A_recv), [0]
    else:
        y_u = y_0

        A_recv = torch.zeros_like(A)
        e_y = y_u

        errors = []    
        for i in range(epoch):
            A_recv_e_y = mismatch_equation(y_0, e_y, A)
            A_recv += A_recv_e_y
            e_y = y_u - A_recv @ PM_image
            e = abs(e_y.cpu().numpy()).mean()
            if verbose:
                print("========PM-error-{}==========".format(i))
                print(e)
            errors.append(e)

        return (y_0, A_recv), errors



def iterative_algo_2(y_u, params, verbose=True):
    A = params["A"]
    PM_image = params["PM_image"]
    PM_epoch = params["PM_epoch"]
    isfloat64 = params["isfloat64"]
    
    (y_0, A_recv), pm_errors = iterative_algo_pm(A, 
                                                 PM_image=PM_image,
                                                 epoch=PM_epoch,
                                                 verbose=verbose,isfloat64=isfloat64)
    
    speckle_measure = params["speckle_measure"]
    epoch = params["epoch"]
    calc_error = params["calc_error"]
    
    
    y = speckle_measure(A_recv)
    y_pm = A_recv @ PM_image
    k = y / y_pm
    print(k)
    
    e_y = y_u - k * y_pm
    
    errors = []    
    start_time = time.time()
    
    for i in range(epoch):
        A_recv_e_y = mismatch_equation(y_0, e_y, A)
        A_recv += A_recv_e_y
        e_y = y_u - k * (A_recv @ PM_image)
        e = abs((y_u - calc_error(A_recv)).cpu().numpy()).mean()
        if verbose:
            print("========error-{}==========".format(i))
            print(e)
        errors.append(e)
    end_time = time.time()
        
    return (A_recv,k), pm_errors, errors, end_time-start_time
    
    
        
    
    
    