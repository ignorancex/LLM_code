import jax 
import jax.numpy as jnp
from functools import partial

from .static_kernels import linear_kernel, rbf_kernel
from ._hyp0f1 import hyp0f1


## TODO: symmetric not implemented yet


class MonomialInterpolationSolver:
    """
    Solver for signature kernel PDE that uses monomial interpolation. 

    Args:
        static_ker (str, optional) : Type of static kernel to use. Supported values:
                                     'linear' or 'rbf'. Defaults to 'linear'.
        scale (float, optional)    : Scaling parameter for the static kernel. Defaults to 1.0.
        order (int, optional)      : Interpolation order. Defaults to 4.
    """
    def __init__(self, 
                 static_ker : str = 'linear',
                 scale : float = 1.0,
                 order : int = 4):
        
        self.deg = order

        # Compute utilities
        self.nodes = self._compute_utils(self.deg)

        if static_ker == 'linear':
            self.static_kernel = lambda x2, x1, y2, y1: linear_kernel(x2, x1, y2, y1, scale)

        elif static_ker == 'rbf':
            self.static_kernel = lambda x2, x1, y2, y1: rbf_kernel(x2, x1, y2, y1, scale)
        

    ########################################################################
    # Utility functions
    ########################################################################

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _compute_utils(deg : int):
        
        # Compute nodes
        nodes = jnp.cos(jnp.pi * (deg - jnp.arange(deg+1)) / (deg))
        
        return (1 + nodes) / 2

    @staticmethod
    def _diag_axis_masks(p : int, length_X : int, length_Y : int):
        
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
    def _initial_conditions(deg : int, dtype):
        ic = jnp.ones(shape=(deg+1), dtype = dtype)
        return ic
    
    @staticmethod
    def _get_idx(k : int):
        idx1 = jnp.where(k % 2 ==0, k-2, k)
        idx2 = k-1
        idx_data = (k + 1) // 2 - 1
        return idx1, idx2, idx_data
    

    ########################################################################
    # Diagonal updates
    ########################################################################

    @partial(jax.jit, static_argnums=(0,))
    def _static_kernel_diag_update(self, 
                                   p: int, 
                                   diag_axis_masked: jnp.ndarray, 
                                   X: jnp.ndarray, 
                                   Y: jnp.ndarray) -> jnp.ndarray:

        return jax.vmap(
                jax.vmap(
                 jax.vmap(lambda i, j, k: jnp.where(k != -1, self.static_kernel(X[i,k+1], X[i,k], Y[j,p-k], Y[j,p-k-1]), 0.0),
                        in_axes=(None, None, 0)),
                 in_axes=(None, 0, None)),
                in_axes=(0, None, None)
               )(jnp.arange(X.shape[0]), jnp.arange(Y.shape[0]), diag_axis_masked)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_interpolation_coeffs(self, 
                                  diag_solution_minus1 : jnp.ndarray, 
                                  diag_axis_mask : jnp.ndarray) -> jnp.ndarray:
        
        empty = jnp.empty(self.deg+1)

        return jax.vmap(
                jax.vmap(
                 jax.vmap(lambda i, j, k: jnp.where(k != -1, jnp.flip(jnp.polyfit(self.nodes, diag_solution_minus1[i,j,k], 
                                                                                  self.deg)), empty),
                        in_axes=(None, None, 0)),
                 in_axes=(None, 0, None)),
                in_axes=(0, None, None)
               )(jnp.arange(diag_solution_minus1.shape[0]), jnp.arange(diag_solution_minus1.shape[1]), diag_axis_mask)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_hyp0f1(self, 
                    data_at_nodes : jnp.ndarray,
                    diag_axis_mask : jnp.ndarray) -> jnp.ndarray:
        
        zeros = jnp.empty((self.deg, self.deg+1))
        values = jnp.arange(1, self.deg+2)
        
        def _get_hyp0f1_single(i : int, j : int, k : int):

            _data_at_nodes = data_at_nodes[i,j,k]

            return jnp.where(k != -1, hyp0f1(values, _data_at_nodes[..., None]), zeros)
        
        return jax.vmap(
                jax.vmap(
                 jax.vmap(_get_hyp0f1_single,  in_axes=(None, None, 0)),
                 in_axes=(None, 0, None)),
                in_axes=(0, None, None)
               )(jnp.arange(data_at_nodes.shape[0]), jnp.arange(data_at_nodes.shape[1]), diag_axis_mask)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_data_and_coeffs(self,
                             p: int,
                             diag_data_axis_mask: jnp.ndarray,
                             diag_sol_mask_old: jnp.ndarray,
                             diag_solution_minus1: jnp.ndarray,
                             X: jnp.ndarray,
                             Y: jnp.ndarray) -> jnp.ndarray:

        # Initialise data diagonals of inner products of increments.
        ker = self._static_kernel_diag_update(p, diag_data_axis_mask, X, Y)
        coeffs = self._get_interpolation_coeffs(diag_solution_minus1, diag_sol_mask_old)

        data_at_nodes = ker[..., None] * self.nodes[1:]

        _hyp0f1 = self._get_hyp0f1(data_at_nodes, diag_data_axis_mask)
    
        i0term = _hyp0f1[..., 0]


        return coeffs, _hyp0f1, i0term
    
    ########################################################################
    # Solution updates
    ########################################################################
    
    @partial(jax.jit, static_argnums=(0))
    def _solution_diag_update(self,
                              p : int,
                              diag_axis_mask : jnp.ndarray,
                              diag_solution_minus1 : jnp.ndarray,
                              coeff : jnp.ndarray,
                              hyp0f1_mat : jnp.ndarray,
                              i0term : jnp.ndarray) -> jnp.ndarray:
        """
        Given the data kernel evaluations for the current diagonal, update the solution 
        along that diagonal.
        """
        ic = self._initial_conditions(self.deg, diag_solution_minus1.dtype)
        zeros = jnp.empty_like(ic)  
        ratio_pow = self.nodes[1:, None] ** jnp.arange(self.deg+1)[None,:] 

        def _solution_single_update(i : int, j : int, k : int):

            idx1, idx2, idx_data = self._get_idx(k)

            prev_bd            = diag_solution_minus1[i, j, idx1]          
            prev_bd_opposite   = diag_solution_minus1[i, j, idx2] 
            _coeff_bd          = coeff[i, j, idx1]
            _coeff_bd_opposite = coeff[i, j, idx2]
            _hyp0f1            = hyp0f1_mat[i, j, idx_data] - 1
            _i0term            = i0term[i, j, idx_data]


            new_bc = jnp.zeros_like(prev_bd)
            new_bc = new_bc.at[0].set(prev_bd_opposite[-1])

            part1 = prev_bd[1:] + prev_bd_opposite[-1] - prev_bd[0] * _i0term

            coeff_pow = jnp.einsum('km, m -> km', ratio_pow, _coeff_bd)
            h1 = jnp.einsum('km, km -> k', _hyp0f1, coeff_pow)
            h2 = jnp.einsum('km, m -> k', _hyp0f1, _coeff_bd_opposite)

            new_bc = new_bc.at[1:].add(part1+h1+h2)
                
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

    @partial(jax.jit, static_argnums=(0,))
    def _solve(self, X : jnp.ndarray, Y : jnp.ndarray) -> jnp.ndarray:
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

        diag_solution_minus1 = jnp.zeros(shape=(batch_X, batch_Y, diag_length, self.deg + 1), dtype = X.dtype) 
        diag_solution_minus1 = diag_solution_minus1.at[..., :].set(1.0)

        diag_axis_mask_solution, _ = self._diag_axis_masks(0, length_X, length_Y) # You can brute force the first step and
    
        def _loop(p, carry):

            diag_solution_minus1, diag_axis_mask_solution = carry
            diag_axis_mask_solution_new, diag_axis_mask_data = self._diag_axis_masks(p, length_X, length_Y)
            
            coeff, _hyp0f1, _i0term = self._get_data_and_coeffs(p, 
                                                                diag_axis_mask_data, 
                                                                diag_axis_mask_solution,
                                                                diag_solution_minus1,
                                                                X, 
                                                                Y)

            diag_solution = self._solution_diag_update(p, 
                                                       diag_axis_mask_solution_new, 
                                                       diag_solution_minus1, 
                                                       coeff,
                                                       _hyp0f1,
                                                       _i0term)

            return diag_solution, diag_axis_mask_solution_new
        
        diag_solutions, _ = jax.lax.fori_loop(1, diag_iterations+1, _loop, (diag_solution_minus1, diag_axis_mask_solution))

        return jnp.mean(diag_solutions[:,:,-2:,-1], axis=-1)
    

    @partial(jax.jit, static_argnums=(0,3,4))
    def solve(self, 
              X: jnp.ndarray, 
              Y: jnp.ndarray, 
              sym : bool = False,
              multi_gpu : bool = False) -> jnp.ndarray:
        """
        Allows for multi-GPU parallelisation of the solver.
        """

        if sym:
            raise NotImplementedError("Symmetric solver not implemented yet.")
        
        # Determine the number of GPUs (if any)
        try:
            num_gpus = jax.lib.xla_bridge.get_backend('gpu').device_count()
        except RuntimeError:
            num_gpus = 0

        # If only one GPU or no GPU is available, or multi_gpu is False, 
        # just run the single-device solver
        if (num_gpus <= 1) or (not multi_gpu):
            return self._solve(X, Y)

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