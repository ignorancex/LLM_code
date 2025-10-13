import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.integrate import solve_ivp
from math import pi, log, sqrt
from copy import deepcopy
import matplotlib.pyplot as plt
from WCs_dic import C_in_down, C_in_up
from beta_function import beta_function, beta_function_twoloopSM

def rge_solve(C_in, scale_in, scale_out, rtol=1e-6, atol=1e-8, SM_loop='one'):
    if SM_loop not in ['one', 'two']:
        raise ValueError("SM_loop must be either 'one' or 'two'.")

    beta_func = beta_function if SM_loop == 'one' else beta_function_twoloopSM

    def fun(t, y):
        C = {k: y[i] for i, k in enumerate(C_in.keys())}
        beta_vals = beta_func(C, t)
        return np.array([
            (np.sum(beta_vals[k])) / (16 * np.pi**2) if isinstance(beta_vals[k], (np.ndarray, list))
            else beta_vals[k] / (16 * np.pi**2)
            for k in C_in.keys()
        ], dtype=np.complex128)

    y0 = np.array([C_in[k] for k in C_in.keys()], dtype=np.complex128)

    sol = solve_ivp(
        fun=fun,
        t_span=(np.log(scale_in), np.log(scale_out)),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol
    )

    C_out = {k: sol.y[i, -1] for i, k in enumerate(C_in.keys())}
    return C_out

def rge_solve_leadinglog(C_in, scale_in, scale_out, SM_loop='one'):
    if SM_loop not in ['one', 'two']:
        raise ValueError("SM_loop must be either 'one' or 'two'.")

    beta_func = beta_function if SM_loop == 'one' else beta_function_twoloopSM
    C_out = deepcopy(C_in)
    beta_vals = beta_func(C_out, None)

    for k, C in C_out.items():
        C_out[k] = C + beta_vals[k] / (16 * np.pi**2) * np.log(scale_out / scale_in)

    return C_out

def solve_RGE(scale_in, scale_out, C_in, method='integrate', rtol=1e-6, atol=1e-8, SM_loop='one'):
    if SM_loop not in ['one', 'two']:
        raise ValueError("SM_loop must be either 'one' or 'two'.")

    if method == 'leadinglog':
        return rge_solve_leadinglog(C_in, scale_in, scale_out, SM_loop=SM_loop)
    elif method == 'integrate':
        return rge_solve(C_in, scale_in, scale_out, rtol=rtol, atol=atol, SM_loop=SM_loop)
    else:
        raise ValueError("Method must be 'leadinglog' or 'integrate'")

def solve_rge(scale_in, scale_out, C_in, method='integrate', basis='down', rtol=1e-6, atol=1e-8, SM_loop='one'):
    if SM_loop not in ['one', 'two']:
        raise ValueError("SM_loop must be either 'one' or 'two'.")

    if scale_in < 79.99999 or scale_out < 79.99999:
        raise ValueError("scale_in and scale_out must be larger than or equal to electroweak scale 80 GeV")

    scale_in_SMpar = 91.1876
    scale_out_SMpar = scale_in

    if basis == 'down':
        C_out_SMpar = solve_RGE(scale_in_SMpar, scale_out_SMpar, C_in_down, method=method, rtol=rtol, atol=atol, SM_loop=SM_loop)
    elif basis == 'up':
        C_out_SMpar = solve_RGE(scale_in_SMpar, scale_out_SMpar, C_in_up, method=method, rtol=rtol, atol=atol, SM_loop=SM_loop)
    else:
        raise ValueError("Invalid basis. Please use 'down' or 'up'.")

    C_out_SMpar.update(C_in)
    C_out = solve_RGE(scale_in, scale_out, C_out_SMpar, method=method, rtol=rtol, atol=atol, SM_loop=SM_loop)
    return C_out

# function to print WCs
#################################################################################################
def print_WCs(C_out, specific_keys=None):
    exclude_keys = {
        "Yu11", "Yu12", "Yu13", "Yu21", "Yu22", "Yu23", "Yu31", "Yu32", "Yu33",
        "Yd11", "Yd12", "Yd13", "Yd21", "Yd22", "Yd23", "Yd31", "Yd32", "Yd33",
        "Yl11", "Yl12", "Yl13", "Yl21", "Yl22", "Yl23", "Yl31", "Yl32", "Yl33",
        "gp", "g", "gs", "Lambda", "muh"
    }
    priority_keys = [
        "LH5_11", "LH5_12", "LH5_13", "LH5_22", "LH5_23", "LH5_33",
        "LH_11", "LH_12", "LH_13", "LH_22", "LH_23", "LH_33"
    ]
    if specific_keys is None:
        keys_to_print = [k for k in C_out.keys() if k not in exclude_keys]
    else:
        keys_to_print = specific_keys if isinstance(specific_keys, list) else [specific_keys]
    nonzero_keys = [k for k in keys_to_print if C_out.get(k, 0) != 0]
    print("## Wilson coefficients\n")
    print("**EFT:** `SMEFT`")
    if nonzero_keys:
        print("| WC name | Value |")
        print("|--------------------|----------------------------------------------------|")

        for key in priority_keys:
            if key in nonzero_keys:
                print(f"| `{key}` | {C_out[key]} |")
                nonzero_keys.remove(key)

        sorted_keys = sorted(nonzero_keys, key=lambda k: abs(C_out[k]), reverse=True)
        for key in sorted_keys:
            value = C_out[key]
            print(f"| `{key}` | {value} |")
    else:
        print("No nonzero Wilson coefficients found.")

# function to plot WC evolution
#################################################################################################
def plot_WC(
    scale_initial, scale_final, C_in_initial, C_yaxis, number_points, method='integrate', basis="down",
    legend_labels=None, xlabel=None, ylabel=None, caption=None,
    xscale='log', yscale='log', rtol=1e-6, atol=1e-8, mode='abs', SM_loop='one'
):
    if SM_loop not in ['one', 'two']:
        raise ValueError("SM_loop must be either 'one' or 'two'.")

    if scale_initial < 79.999999999 or scale_final < 79.999999999:
        raise ValueError("scale_initial and scale_final must be larger than or equal to electroweak scale 80 GeV")

    if mode not in ['abs', 'real']:
        raise ValueError("mode must be either 'abs' (absolute value) or 'real' (real part).")

    wilson_keys = list(C_yaxis)
    wilson_evolution = {key: [] for key in wilson_keys}
    scales = np.logspace(np.log10(scale_initial), np.log10(scale_final), num=number_points)

    for scale in scales:
        C_out = solve_rge(scale_initial, scale, C_in_initial, method=method, basis=basis, rtol=rtol, atol=atol, SM_loop=SM_loop)
        for key in wilson_keys:
            value = C_out.get(key, 0)
            wilson_evolution[key].append(np.abs(value) if mode == 'abs' else np.real(value))

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 7))

    for key in wilson_keys:
        label = legend_labels[key] if legend_labels and key in legend_labels else key
        plt.plot(scales, wilson_evolution[key], label=label, linewidth=2)

    plt.xscale(xscale)
    plt.yscale(yscale)

    xlabel = xlabel if xlabel else r'Energy Scale $\Lambda$ (GeV)'
    ylabel = ylabel if ylabel else (r'$|C_{d}^i({\Lambda})|\ [\text{GeV}^{4-d}]$' if mode == 'abs' else r'$\Re(C_{d}^i({\Lambda}))\ [\text{GeV}^{4-d}]$')
    caption = caption if caption else (r'Wilson Coefficients Evolution (Absolute Values)' if mode == 'abs' else r'Wilson Coefficients Evolution (Real Part)')

    plt.xlabel(xlabel, fontsize=24, labelpad=2)
    plt.ylabel(ylabel, fontsize=24, labelpad=5)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title(caption, fontsize=20, pad=2)
    plt.legend(fontsize=20, loc='best', frameon=True)
    plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
    plt.show(block=False)









