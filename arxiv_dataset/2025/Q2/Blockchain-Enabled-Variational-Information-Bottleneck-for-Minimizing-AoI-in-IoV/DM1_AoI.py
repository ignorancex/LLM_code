import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# 定义常数
M = 3
tau_c = 0.003
p_d = 0.02
D = 0.6

# 生成 rho 的值
rho_values = np.linspace(0.2, 0.8, 100)
# mu = 1 / (D * rho_values)
# 计算 Delta 的值
def calculate_delta(rho):
    mu = 1 / (D * rho)
    beta = -rho * lambertw(-rho ** (-1) * np.exp(-1/rho))
    term1 = 1
    # term2 = 1 + 1 / (2 * rho) + beta / (1 - beta)
    term2 = 1 + 1 / (2 * rho) + beta / (1 - beta) * np.exp(-D * M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

delta_values = [calculate_delta(rho) for rho in rho_values]

# 绘图
plt.figure()
plt.plot(rho_values, delta_values)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Delta$')
plt.grid(True)
plt.show()
