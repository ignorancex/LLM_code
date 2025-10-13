import numpy as np
import matplotlib.pyplot as plt

# 定义常数
mu = 0.3
M = 3
tau_c = 0.003
p_d = 0.02

# 生成 rho 的值
rho_values = np.linspace(0.2, 0.8, 100)

# lambda_values = rho_values * mu

# 计算 Delta 的值
def calculate_delta(rho):
    term1 = 1 / mu
    lambda_values = rho * mu
    term2 = 1 + 1 / rho + (rho**2) / (1 - rho) * np.exp(-lambda_values* M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

delta_values = [calculate_delta(rho) for rho in rho_values]

# 绘图
plt.plot(rho_values, delta_values)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Delta$')
# plt.title('Delta vs. rho')
plt.grid(True)
plt.show()
