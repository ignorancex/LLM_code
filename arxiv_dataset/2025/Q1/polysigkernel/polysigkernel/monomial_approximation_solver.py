import jax 
import jax.numpy as jnp
from functools import partial
from .static_kernels import linear_kernel, rbf_kernel

class MonomialApproximationSolver:
    """
    Solver for signature kernel PDE that uses monomial approximation. 

    Args:
        static_ker (str, optional) : Type of static kernel to use. Supported values:
                                     'linear' or 'rbf'. Defaults to 'linear'.
        scale (float, optional)    : Scaling parameter for the static kernel. Defaults to 1.0.
        order (int, optional)      : Monomial expansion order. Defaults to 4.
    """
    def __init__(self, 
                 static_ker : str = 'linear',
                 scale : float = 1.0,
                 order : int = 4):
        
        self.order = order

        # Precompute utility matrices (mat1 and mat2) used for monomial-based updates.
        self.mat1, self.mat2 = self._compute_utils(order)

        # Initialize the desired static kernel (linear or RBF) with the given scale.
        if static_ker == 'linear':
            self.static_kernel = lambda x2, x1, y2, y1: linear_kernel(x2, x1, y2, y1, scale)

        elif static_ker == 'rbf':
            self.static_kernel = lambda x2, x1, y2, y1: rbf_kernel(x2, x1, y2, y1, scale)
        

    ########################################################################
    # Utility functions
    ########################################################################

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _compute_utils(order : int):
        """
        Compute utility matrices for monomial expansions up to a given order.
        """
        n_range = jnp.arange(1, order+1)[:, None]
        k_range = jnp.arange(order+1)

        factorials = jnp.cumprod(jnp.arange(1, 2*order+1))
        factorials = jnp.concatenate([jnp.array([1.0]), factorials])

        k_over_n = factorials[k_range[None, :]] / factorials[n_range[:order+1]]

        mat1 = jnp.where(jnp.tril(jnp.ones((order-1, order-1))), k_over_n[1:,1:-1] / jnp.tril(factorials[n_range[1:] - k_range[1:-1]]), 0.0)
        mat2 = k_over_n / factorials[n_range + k_range[None, :]]

        return mat1, mat2
    

    @staticmethod
    def _diag_axis_masks(p : int, length_X : int, length_Y : int):
        """
        Generate masks for indexing the diagonal elements in both the solution array 
        and the data array (which is used to compute the kernel evaluations).
        """
        
        diag_length_solution = 2 * (length_X - 1) 
        diag_length_data = length_X - 1

        diag_axis_solution = jnp.arange(diag_length_solution)
        diag_axis_data = jnp.arange(diag_length_data)

        start_row_solution = jnp.where((p == length_X + length_Y - 3), jnp.maximum(0, 2 * (p - length_Y + 1)), jnp.maximum(0, 2 * (p - length_Y + 2)))
        end_row_solution = jnp.minimum(diag_length_solution, 2 * p + 2)

        start_row_data = jnp.maximum(0, p - length_Y + 1)
        end_row_data = jnp.minimum(diag_length_data, p)

        mask_solution = jnp.where((diag_axis_solution >= start_row_solution) & (diag_axis_solution < end_row_solution), diag_axis_solution, -1)
        mask_data = jnp.where((diag_axis_data >= start_row_data) & (diag_axis_data < end_row_data), diag_axis_data, -1)    

        return jnp.where((p == length_X + length_Y - 3), mask_solution.at[-2:].add(1), mask_solution), mask_data
    

    @staticmethod
    def _initial_conditions(order : int, dtype):
        ic = jnp.zeros(shape=(order+1), dtype = dtype)
        return ic.at[0].set(1.0)
    
    @staticmethod
    def _get_idx(k : int):
        """
        For a given diagonal offset k, compute the indices needed to access 
        the correct elements from the solution and data arrays.
        """
        idx1 = jnp.where(k % 2 ==0, k-2, k)
        idx2 = k-1
        idx_data = (k + 1) // 2 - 1
        return idx1, idx2, idx_data
    

    ########################################################################
    # Diagonal updates
    ########################################################################

    @partial(jax.jit, static_argnums=(0, 5))
    def _get_diag_data(self, 
                       p: int, 
                       diag_axis_masked: jnp.ndarray, 
                       X: jnp.ndarray, 
                       Y: jnp.ndarray,
                       sym : bool = False) -> jnp.ndarray:
        
        def _get_kernel(i : int, j : int, k : int):
            if sym:
                return jnp.where((k != -1) & (i >= j), self.static_kernel(X[i,k+1], X[i,k], Y[j,p-k], Y[j,p-k-1]), 0.)
            else:
                return jnp.where(k != -1, self.static_kernel(X[i,k+1], X[i,k], Y[j,p-k], Y[j,p-k-1]), 0.)


        return jax.vmap(
                jax.vmap(
                 jax.vmap(_get_kernel, in_axes=(None, None, 0)),
                 in_axes=(None, 0, None)),
                in_axes=(0, None, None)
               )(jnp.arange(X.shape[0]), jnp.arange(Y.shape[0]), diag_axis_masked)
    

    ########################################################################
    # Solution updates
    ########################################################################
    
    @partial(jax.jit, static_argnums=(0,5))
    def _solution_diag_update(self,
                              p : int,
                              diag_axis_mask : jnp.ndarray,
                              diag_solution_minus1 : jnp.ndarray,
                              diag_data : jnp.ndarray,
                              sym : bool = False):
        """
        Given the data kernel evaluations for the current diagonal, update the solution 
        along that diagonal.
        """
        mat1 = self.mat1
        mat2 = self.mat2

        ic = self._initial_conditions(self.order, diag_data.dtype)
        zeros = jnp.empty_like(ic) 

        def _solution_single_update(i : int, j : int, k : int):

            idx1, idx2, idx_data = self._get_idx(k)

            prev_bd          = diag_solution_minus1[i, j, idx1]         
            prev_bd_opposite = diag_solution_minus1[i, j, idx2]
            ker              = diag_data[i, j, idx_data]                     

            ker_powers = jnp.power(ker, jnp.arange(1,self.order+1))
            ker_powers_reversed = jnp.tril(jax.scipy.linalg.toeplitz(ker_powers[:-1]))  

            new_bc = prev_bd.at[0].set(jnp.sum(prev_bd_opposite))
            new_bc = new_bc.at[1:].add(jnp.dot(mat2, prev_bd_opposite) * ker_powers)
            new_bc = new_bc.at[2:].add(jnp.dot(mat1 * ker_powers_reversed, prev_bd[1:-1]))

            # If sym=True, only update when i >= j.
            if sym:
                return jnp.where((k != -1) & (i >= j), jnp.where((k == 0) | (k == 2*p+1), ic, new_bc), zeros)
            else:
                return jnp.where(k != -1, jnp.where((k == 0) | (k == 2*p+1), ic, new_bc), zeros)
        
        return jax.vmap(
                jax.vmap(
                 jax.vmap(_solution_single_update, in_axes=(None, None, 0)),
                 in_axes=(None, 0, None)),
                in_axes=(0, None, None)
               )(jnp.arange(diag_solution_minus1.shape[0]), 
                 jnp.arange(diag_solution_minus1.shape[1]), 
                 diag_axis_mask)



    ########################################################################
    # Main solvers
    ########################################################################

    @partial(jax.jit, static_argnums=(0,3))
    def _solve(self, X : jnp.ndarray, Y : jnp.ndarray, sym : bool = False) -> jnp.ndarray:
        """
        The core solver method, which iterates over diagonals.

        Args:
            X (jax.numpy.ndarray): 3D array with shape (batch_X, length_X, dim_X).
            Y (jax.numpy.ndarray): 3D array with shape (batch_Y, length_Y, dim_Y).
            sym (bool, optional): If True, enforce symmetry by only updating i >= j.
                                  The final result is combined symmetrically. Defaults to False.

        Returns:
            jax.numpy.ndarray: A 2D solution array of shape (batch_X, batch_Y) containing 
                               the final aggregated results of the monomial expansions.
        """
        # Problem sizes
        batch_X, batch_Y = X.shape[0], Y.shape[0]
        length_X, length_Y = X.shape[1], Y.shape[1]
        diag_length = 2 * (length_X - 1)                #(jnp.maximum(length_X, length_Y) - 1) * 2
        diag_iterations = length_X + length_Y - 3    

        diag_solution_minus1 = jnp.zeros(shape=(batch_X, batch_Y, diag_length, self.order + 1), dtype = X.dtype).at[..., 0].set(1.0) 
        #diag_solution_minus1 = diag_solution_minus1
    
        def _loop(p, carry):

            diag_solution_minus1 = carry

            diag_axis_mask_solution, diag_axis_mask_data = self._diag_axis_masks(p, length_X, length_Y)
            diag_data = self._get_diag_data(p, diag_axis_mask_data, X, Y, sym)

            diag_solution = self._solution_diag_update(p, 
                                                       diag_axis_mask_solution, 
                                                       diag_solution_minus1, 
                                                       diag_data,
                                                       sym) 

            return diag_solution
        
        diag_solutions = jax.lax.fori_loop(1, diag_iterations+1, _loop, (diag_solution_minus1))

        solution = jnp.mean(jnp.sum(diag_solutions[:,:,-2:,:], axis=-1), axis=-1)

        if sym:
            return solution + solution.swapaxes(0,1) - jnp.diag(jnp.diag(solution)) 
        else:
            return solution

    
    @partial(jax.jit, static_argnums=(0,3,4))
    def solve(self, 
              X: jnp.ndarray, 
              Y: jnp.ndarray, 
              sym : bool = False,
              multi_gpu : bool = False) -> jnp.ndarray:
        """
        Allows for multi-GPU parallelisation of the solver.
        """
        
        # Determine the number of GPUs (if any)
        try:
            num_gpus = jax.lib.xla_bridge.get_backend('gpu').device_count()
        except RuntimeError:
            num_gpus = 0

        # If only one GPU or no GPU is available, or multi_gpu is False, 
        # just run the single-device solver
        if (num_gpus <= 1) or (not multi_gpu):
            return self._solve(X, Y, sym)

        # Find the largest integer "num_parallel" <= "total" that is divisible by num_gpus
        total        = X.shape[0]
        num_parallel = (total // num_gpus) * num_gpus
        remaining    = total - num_parallel

        # Parallise solver across GPUs
        X_parallel = X[:num_parallel]
        X_sub_tensors = jnp.stack(jnp.array_split(X_parallel, num_gpus))
        Z_sub_tensors = jax.pmap(self._solve, in_axes=(0, None))(X_sub_tensors, Y)
        Z = jnp.concatenate(Z_sub_tensors, axis=0)

        # If all the data has been used just return it
        if remaining == 0:
            return Z

        # Otherwise perform the final computation on a single GPU and concatenate with the rest
        X_remainder = X[num_parallel:]
        Z_remainder = self._solve(X_remainder, Y)

        return jnp.concatenate([Z, Z_remainder], axis=0)