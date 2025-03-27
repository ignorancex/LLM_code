import torch
import numpy as np
import matplotlib.pyplot as plt
MAX_DENSITY_MISMATCH = 0.01

def identity(n, m):
    I = np.identity(min(n, m), dtype=float)
    I = torch.Tensor(I)
    if n > m: # append zero rows to match the desired size n x m
        I = torch.cat((I, torch.zeros((n - m, m))), dim=0)
    if n < m: # append zero columns to match the desired size n x m
        I = torch.cat((I, torch.zeros((n, m - n))), dim=1)
    return I

def rand_givens(n):
    i = np.random.randint(0, n)
    j = i
    while j == i:
        j = np.random.randint(0, n)
    if j < i:
        tmp = i
        i = j
        j = tmp
    alpha = np.random.uniform(0.0, 2*np.pi)
    return i, j, alpha

def mul_by_givens(A, i, j, alpha):
    A[:, [i, j]] = torch.cat((
        np.cos(alpha) * A[:, [i]] + np.sin(alpha) * A[:, [j]],
       -np.sin(alpha) * A[:, [i]] + np.cos(alpha) * A[:, [j]],
    ), dim=1)

def get_density(a):
    n = len(a)
    m = len(a[0])
    non_zero = torch.count_nonzero(a)
    return non_zero.item() / (n * m)

def orthogonal_with_density(n, m, d):
    transposed = False
    if m < n:
        transposed = True
        tmp = n
        n = m
        m = tmp
    elif n < m:
        print('Warning: n < m, produced matrix will be row-orthogonal:', n, m)
    if d > 1. or d < 0.:
        print(f'Warning: density should belong to [0, 1], found {round(d, 6)}')
        return identity(n, m)
    A = identity(n, m)
    prev = identity(n, m)

    # if get_density(A) > d:
    #     raise Exception('Impossible to obtain such density {} for matrix of shape {}')
        
    iters = 0
    while get_density(A) < d:
        prev = A.clone()
        i, j, alpha = rand_givens(m)
        mul_by_givens(A, i, j, alpha)
        iters += 1

    if abs(d - get_density(prev)) < abs(d - get_density(A)):
        A = prev
        
    print(f'Produced orthogonal matrix with density {round(get_density(A), 6)}, asked for {round(d, 6)}, after {iters} iters')
    if transposed:
        A = torch.transpose(A, 0, 1)
    return A

if __name__ == '__main__':
    # print(orthogonal_with_density(10, 20, 0.99))
    # print(orthogonal_with_density(20, 10, 0.99))
    orthogonal_with_density(100, 100, 0.14)
    orthogonal_with_density(100, 100, -0.1)
    orthogonal_with_density(100, 100, 1.1)
    orthogonal_with_density(100, 100, 0.5)
    orthogonal_with_density(100, 100, 0.99)
    ''' Outputs something like:
    Got A with density 0.1404, asked for 0.14, after 156 iters
    Warning: density should belong to [0, 1], found -0.1
    Warning: density should belong to [0, 1], found 1.1
    Got A with density 0.5029, asked for 0.5, after 249 iters
    Got A with density 0.9904, asked for 0.99, after 510 iters
    '''