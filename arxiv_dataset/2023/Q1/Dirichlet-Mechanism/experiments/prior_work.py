import quadpy

from mpmath import (beta,
                    gamma,
                    log,
                    re,
                    zeros)


def compute_vertices(e1, e2):
    v_list = []
    if 1 - 2 * e1 > e2:
        v0 = zeros(1, 3)
        v0[0] = e1
        v0[1] = e1
        v0[2] = 1 - 2 * e1
        v_list.append(v0)

    if 1 - e1 - e2 > e1:
        v1 = zeros(1, 3)
        v1[0] = e1
        v1[1] = e2
        v1[2] = 1 - e1 - e2
        v_list.append(v1)

    return v_list


def compute_vertices_binary(e1, e2):
    if 1 - e1 > e2:
        v = zeros(1, 2)
        v[0] = e1
        v[1] = 1 - e1 - e2

    return v


def compute_delta(g, k, e1, e2, int_scheme):
    scheme = quadpy.t2.schemes[int_scheme]()
    v_list = compute_vertices(e1, e2)
    probs = [0]*len(v_list)
    for i, v in enumerate(v_list):
        def f(x):
            return ((x[0]**(k*v[0] - 1)) *
                    (x[1]**(k*v[1] - 1)) *
                    ((1 - x[0] - x[1])**(k*v[2] - 1)))
        integral = scheme.integrate(f, [[g, g],
                                        [1-2*g, g],
                                        [g, 1-2*g]])
        Z = gamma(k*v[0])*gamma(k*v[1])*gamma(k*v[2])
        Z /= gamma(k*sum(v))
        probs[i] = integral / Z

    delta = 1 - min(probs)

    return re(delta)


def compute_delta_binary(g, k, e1, e2):
    scheme = quadpy.c1.gauss_patterson(5)
    v = compute_vertices_binary(e1, e2)

    def f(x):
        return ((x[0]**(k*v[0] - 1)) *
                ((1 - x[0])**(k*v[1] - 1)))
    if g < 0.5:
        integral = scheme.integrate(f, [[g], [1-g]])
    else:
        integral = scheme.integrate(f, [[1-g], [g]])
    Z = gamma(k*v[0])*gamma(k*v[1])
    Z /= gamma(k*sum(v))
    probs = integral / Z

    delta = 1 - probs

    return re(delta)


def compute_epsilon(g, k, W, e1, e2, ss):
    log_arg1 = beta(k * e1, k * (1 - e1 - e2))
    log_arg2 = beta(k * (e1 + ss/2), k * (1 - e1 - e2 - ss/2))
    log_arg3 = 1 / g - W + 1
    epsilon = log(log_arg1 / log_arg2) + 0.5 * k * ss * log(log_arg3)
    return re(epsilon)
