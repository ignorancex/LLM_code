import jax
import jax.numpy as jnp
from jax import lax

#############################################
# Hypergeometric function 0F1 implementation
#############################################

DOUBLE_PRECISION = 1e-15

def _hyp_0f1_serie(a,x):
      
    def body(state):
        serie, k, term = state
        serie += term
        term *= x / (k + 1)  / (a + k)
        k += 1
        return serie, k, term
    
    def cond(state):
        serie, k, term = state
        return (k < 250) & (lax.abs(term) / lax.abs(serie) > DOUBLE_PRECISION)
    
    init = 1, 1, x / a

    return lax.while_loop(cond, body, init)[0]

# TODO - implement this 
def _hyp_0f1_asymptotic(a,x):
    return jnp.inf


@jax.jit
@jnp.vectorize
def hyp0f1(a, x):
    """
    Implements the hypergeometric function 0F1 in jax using lax backend.
    """  
    result = lax.cond(lax.abs(x) < 100, _hyp_0f1_serie, _hyp_0f1_asymptotic, a, x)
    index = (a == 0) * 1 

    return lax.select_n(index, result, jnp.array(jnp.inf, dtype=x.dtype))