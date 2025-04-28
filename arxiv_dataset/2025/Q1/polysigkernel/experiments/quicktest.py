import jax
import jax.numpy as jnp
import random
import timeit
import argparse

from polysigker.sigkernel import SigKernel as polysigkernel
from sigkerax.sigkernel import SigKernel as SigKerax
import signax 



def signax_kernel(X, Y, level : int):
    sigs_X = signax.signature(X, level)
    sigs_Y = signax.signature(Y, level)
    return 1. + jnp.einsum("xa,ya -> xy", sigs_X, sigs_Y)


def mean_absolute_percentage_error(ker1, ker2):
    return jnp.mean(jnp.abs((ker1-ker2) / ker2))

models =[
    'monomial_approx',
    'monomial_interp',
    'monomial_approx1',
    'monomial_interp1',
    'monomial_approx2',
    'monomial_interp2',
    'monomial_approx3',
    'cheb_interp'
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_time', help='True (default) to compute the time taken for the analysis, False  otherwise.', action='store_false')
    parser.add_argument('-s', '--solver', type=str, default='monomial_approx', nargs='+', help='Solver to use for the analysis.')
    parser.add_argument('--sigkerax', help='True (default) to compare the SigKernel with the SigKerax, False otherwise.', action='store_false')
    parser.add_argument('--data', type=str, default='bm', choices=['bm', 'smooth'], help='Data to use for the analysis.')


    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data.')
    parser.add_argument('--length', type=int, default=50, help='Length of the time series.')
    parser.add_argument('--dim', type=int, default=2, help='Dimension of the time series.')
    parser.add_argument('--kernel', type=str, default='linear', help='Static kernel to use for the analysis.')
    parser.add_argument('-o', '--order', type=int, default=10, help='Order of the polynomial signature kernel.')
    parser.add_argument('--scale', type=float, default=1e-1)
    parser.add_argument('--normalize', help='True to normalize the data, False otherwise.', action='store_true')

    parser.add_argument('--number', type=int, default=3, help='Number fo runs in a repeat loop')
    parser.add_argument('--repeat', type=int, default=3, help='Number of repeat loops')
    parser.add_argument('--sym', help='True to use the symmetric version of the kernel, False otherwise.', action='store_true')

    parser.add_argument('--print', help='True to print the first 5x5 elements of the kernel matrix, False otherwise.', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu', 'multi_gpu'], help='Device to run the code.')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'], 
                        help='Data type for the data.')
    
    args = parser.parse_args()

    if isinstance(args.solver, str):
        args.solver = [args.solver]
    for solv in args.solver:
        if solv not in models:
            raise ValueError("Solver {} not supported.".format(solv))

    dtype = jnp.float32 if args.dtype == 'float32' else jnp.float64

    if dtype == jnp.float64:
        jax.config.update('jax_enable_x64', True) # Enable 64-bit precision

    def timer_func(func):
        timer = timeit.Timer(lambda : func())
        results = timer.repeat(number=args.number, repeat=args.repeat)
        elapsed_time = sum(results)/ args.repeat / args.number
        return elapsed_time
    

    # Generate data
    key1 = jax.random.PRNGKey(280201)
    key2 = jax.random.PRNGKey(280202)
    
    if args.data == 'bm':
        
        X = args.scale * jax.random.normal(key1, shape=(args.batch_size, args.length, args.dim), dtype=dtype).cumsum(axis=1)
        Y = args.scale * jax.random.normal(key2, shape=(args.batch_size, args.length, args.dim), dtype=dtype).cumsum(axis=1)

    elif args.data == 'smooth':

        X = args.scale * jnp.sin(jax.random.normal(key1, shape=(args.batch_size, args.length, args.dim), dtype=dtype))
        Y = args.scale * jnp.cos(jax.random.normal(key2, shape=(args.batch_size, args.length, args.dim), dtype=dtype))

    if args.normalize:
        X = X / X.max()
        Y = Y / Y.max()
        print(X.max(), Y.max())

    # X = jnp.ones((args.batch_size, args.length, args.dim)) 
    # X = X.at[:,::2,:].set(-1) - jax.random.normal(key1, shape=(args.batch_size, args.length, args.dim), dtype=jnp.float64) * 1e-1
    # Y = jnp.ones((args.batch_size, args.length, args.dim))
    # Y = Y.at[:,::2,:].set(-1) - jax.random.normal(key2, shape=(args.batch_size, args.length, args.dim), dtype=jnp.float64) * 1e-1
    # X = X[...,None] / args.scale
    # Y = Y[...,None] / args.scale
    # print(X[0])

    # X = X / args.scale
    # Y = Y / args.scale

    # print(X.shape, Y.shape)

    # X = X[4,...][None, ...]

    # Y = Y[4,...][None, ...]
    # if args.print:
    #     print(X)
    #     print(Y)
    #     print('-'*50)

    # if args.sym:
    #     Y = X

    # Y = X

    # for l in range(1, args.length):
    #     X = X.at[:,l,:].set(X[:,l-1,:]+1.)
    #     Y = Y.at[:,l,:].set(Y[:,l-1,:]+1.)

    #Truncated sigkernel 
    signax_ker = signax_kernel(X, Y, 21)

    if args.print:
        print(signax_ker[:5,:5])
        print('-'*50)


    for solv in args.solver:

        SK = polysigkernel(order=args.order, static_kernel=args.kernel, solver=solv, add_time=False, scales=jnp.array([1.]))
        sigker_ker = SK.kernel_matrix(X,Y, sym=False, max_batch=100)

        error = mean_absolute_percentage_error(sigker_ker, signax_ker)
        print("Mean Absolute Percentage Error {}: ".format(solv), error)

        if args.compute_time:
            print("Time taken for the SigKernel {}: ".format(solv), timer_func(lambda : SK.kernel_matrix(X, Y)))

        if args.print:
            print(sigker_ker[:5,:5])
            print('='*50)   

    if args.sigkerax:
        
        SK = SigKerax(static_kernel_kind=args.kernel, refinement_factor=args.order, add_time=False)
        sigker_ker = SK.kernel_matrix(X, Y)[...,0]

        error = mean_absolute_percentage_error(sigker_ker, signax_ker)
        print("Mean Absolute Percentage Error Sigkerax: ", error)

        if args.compute_time:
            print("Time taken for the SigKerax: ", timer_func(lambda : SK.kernel_matrix(X, Y)))

        if args.print:
            print(sigker_ker[:5,:5])

        
