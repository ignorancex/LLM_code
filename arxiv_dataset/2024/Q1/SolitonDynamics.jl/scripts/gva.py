import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def sigma_check(eta, g, V0, kL):
    return g * np.sqrt(2/np.pi) + 2 * eta >= 0

def energy(eta, g, V0, kL):
    return -np.exp(-kL**2 * eta**2) * V0  \
    + 1 / np.sqrt(1 + g / (np.sqrt(2 * np.pi) * eta)) \
    + 1 / (4 * eta**2) \
    + g / (np.pi**(1/4) * np.sqrt(eta * (np.sqrt(2) * g + 2 * np.sqrt(np.pi) * eta)))
 
def first_derivative(eta, g, V0, kL):
    if eta <= 0:
        return np.nan
    return - (1 \
           - 4 * np.exp(-kL**2 * eta**2) * kL**2 * V0 * eta**4 \
           + (g * eta**(3/2)) / (np.pi**(1/4) * np.sqrt(np.sqrt(2) * g + 2 * np.sqrt(np.pi) * eta))) \
           / (2 * eta**3)
           

def second_derivative(eta, g, V0, kL):
    return -2 * np.exp(-kL**2 * eta**2) * kL**2 * V0 * (-1 + 2 * kL**2 * eta**2) \
          + (6 + (g * eta**(3/2) * (3 * np.sqrt(2) * g + 8 * np.sqrt(np.pi) * eta)) \
           / (np.pi**(1/4) * (np.sqrt(2) * g + 2 * np.sqrt(np.pi) * eta)**(3/2))) / (4 * eta**4)


def count_zeros(foo1, foo2, g, V0, kL,
                eta_min=1e-2,
                eta_max=10,
                num_points=200):
    # eta_vals = np.linspace(eta_min, eta_max, num=num_points)
    # base = np.exp(1)
    eta_vals = np.logspace(np.log(eta_min), np.log(eta_max), num=num_points, base=np.exp(1))
    # deta3 = eta_vals[5]
    # print(eta_vals)
    # if any([not sigma_check(x, g, V0, kL) for x in eta_vals]):
    #   return 0
    y_vals = [foo1(eta, g, V0, kL) for eta in eta_vals]
    single_site_length = np.pi/kL
    # print(single_site_length)
    zero_crossings = []
    for i in range(len(eta_vals) - 1):
        if np.isnan(y_vals[i]) or np.isnan(y_vals[i+1]):
            continue
        if y_vals[i] * y_vals[i+1] < 0:
            try:
                root = brentq(foo1, eta_vals[i], eta_vals[i+1], args=(g, V0, kL))
                # if not any(np.isclose(root, z, atol=1e-3) for z in zero_crossings):
                zero_crossings.append(root)
            except ValueError:
                pass
        if len(zero_crossings)>3:
          break
    # minima = [x for x in zero_crossings if foo1(x*1.01, g, V0, kL)>0 and foo1(x*0.99, g, V0, kL)<0]
    # # print(zero_crossings)
    # print(minima)
    # if any(x < 1 for x in minima):
    #   return len(minima) + 4
    # else:
    #   return len(minima)
        # if len(zero_crossings) == 2 and zero_crossings[1]<1:
    
    # METHOD 2
    # print(zero_crossings)
    hessians = [foo2(x, g, V0, kL) for x in zero_crossings]
    minima = [x for i, x in enumerate(zero_crossings) if (hessians[i]>0)]

    print(minima)
    value = len(minima)
    if len(minima)>0 and minima[0]<single_site_length:
      value *= -1
    for i, zz in enumerate(minima):
      print(f"{zz:>7.1e}", end="")
      print(f" {hessians[i]:>6.0e} | ", end="")
    print()
    return value
    
    # # METHOD 3
    # d2 = lambda x: foo2(x, g, V0, kL)
    # value = len(zero_crossings)
    # n_zeros = len(zero_crossings) 
    
    # assert(n_zeros<=4)
    # if n_zeros > 0:
    #   # print("first zero crossing: ", zero_crossings[0])
    #   print(f"g = {g:>6.2e}, V0 = {V0:>3.2e}, kL = {kL:>6.2e} >> ", end="") 
    #   for i in range(n_zeros):
    #     print(f"{np.sign(d2(zero_crossings[i])):>5.0f}", end="")
    #   print()
    #   # assert(d2(zero_crossings[0])>0)
    # if any(x<1 for x in zero_crossings):
    #   # the first minimum is an actual minimum
    #   if len(zero_crossings) == 2:
    #     if d2(zero_crossings[1])>0:
    #       if zero_crossings[1]<1:
    #         return value * -1
    #       else:
    #         return 7
    #     else:
    #       return 8
    return len(zero_crossings)
  
    
def count_peaks(foo1, g, V0, kL, eta_min=1e-20, eta_max=5, num_points=1000):
    eta_vals = np.linspace(eta_min, eta_max, num_points)
    newfoo = np.vectorize(lambda x: foo1(x, g, V0, kL))
    y_vals = newfoo(eta_vals)
    minima, _ = find_peaks(-y_vals, distance=int(round(0.05/eta_max*num_points)))
    minima = minima[1:]
    print(eta_vals[minima])
    single_site_length = np.pi/kL
    if any(x<single_site_length for x in eta_vals[minima]):
      return len(minima) * -1
    return len(minima)


def count_minima(foo1, g, V0, kL, eta_min=1e-20, eta_max=4, num_points=5):
    eta_vals = np.linspace(eta_min, eta_max, num_points)    
    found_minima = []

    for x0 in eta_vals:
        result = minimize(foo1, x0, args=(g, V0, kL), bounds=[(eta_min, eta_max)])
        if result.success:
            x_min = result.x[0]
            # Add only unique minima
            if not any(np.isclose(x_min, m, atol=1e-3) for m in found_minima):
                found_minima.append(x_min)
    found_minima = found_minima[1:]
    print(found_minima)
    single_site_length = np.pi/kL
    if any(x < single_site_length for x in found_minima):
      return len(found_minima) * -1
    return len(found_minima)
  
# if __name__== "__main__":
#   def update_plot(g, V0, kL):
#     eta_vals = np.linspace(2e-1, 5, 1000)
#     y_vals = [energy(eta, g, V0, kL) for eta in eta_vals]
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(eta_vals, y_vals)
#     plt.xlabel(r"$\eta$")
#     plt.ylabel(r"$E(\eta)$")
#     plt.title(f"Energy Plot for g={g}, V0={V0}, kL={kL}")
#     plt.grid(True)
#     plt.show()

# # Create sliders for g, V0, and kL
# g_slider = FloatSlider(value=-8.32e-1, min=-10, max=10, step=0.01, description="g:")
# V0_slider = FloatSlider(value=1.53, min=0, max=5, step=0.01, description="V0:")
# kL_slider = FloatSlider(value=2.5, min=0, max=10, step=0.1, description="kL:")

# # Use interact to update the plot
# interact(update_plot, g=g_slider, V0=V0_slider, kL=kL_slider)
#   # # plot an exmaple energy
#   # g = -8.32e-1
#   # V0 = 1.53
#   # kL = 2.5
#   # eta_vals = np.linspace(2e-1, 5, 1000)
#   # y_vals = [energy(eta, g, V0, kL) for eta in eta_vals]
#   # plt.plot(eta_vals, y_vals)
#   # plt.xlabel(r"$\eta$")
#   # plt.ylabel(r"$E(\eta)$")
#   # plt.show()