import jax

# computes log(dist_a(z)/dist_b(z))
# where dist_b is assumed to be constant
# it does this by using the renormalized divided
# version for the gradient wrt z, but returns the
# gradients for dist_a.log_prob(z) wrt dist_a!
# we do this because otherwise the gradient wrt z would become
# unstable if dist_a == dist_b and z is far from the origin
@jax.custom_vjp
def log_prob_div(dist_a, dist_b, dist_ab, z):
    return dist_a.log_prob(z) - dist_b.log_prob(z)

def log_div_grad_fwd(dist_a, dist_b, dist_ab, z):
    primals = dist_a.log_prob(z) - dist_b.log_prob(z)
    return primals, (dist_a, dist_b, dist_ab, z)

def log_div_grad_bkw(res, g):
    dist_a, dist_b, dist_ab, z = res

    # compute different vjps for z, dist_a (none for dist_b, dist_ab)
    _, z_vjp = jax.vjp(lambda d, z: d.log_prob(z), dist_ab, z)
    _, z_grad = z_vjp(g)

    _, a_vjp = jax.vjp(lambda d, z: d.log_prob(z), dist_a, z)
    dist_a_grad, _ = a_vjp(g)

    # return gardients for each input
    return dist_a_grad, None, None, z_grad

log_prob_div.defvjp(log_div_grad_fwd, log_div_grad_bkw)