import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# 定义常数
mu = 1
M = 3
# tau_c = 0.003
rho = 0.53
p_d = 0.02
D = 0.6

# 生成 rho 的值
tauc_values = np.linspace(0.001, 0.008, 100)

def calculate_deltamm1(tau_c):
    term1 = 1 / mu
    lambda_values = rho * mu
    term2 = 1 + 1 / rho + (rho**2) / (1 - rho) * np.exp(-lambda_values* M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

def calculate_deltadm1(tau_c):
    mu = 1 / (D * rho)
    beta = -rho * lambertw(-rho ** (-1) * np.exp(-1/rho))
    term1 = 1
    # term2 = 1 + 1 / (2 * rho) + beta / (1 - beta)
    term2 = 1 + 1 / (2 * rho) + beta / (1 - beta) * np.exp(-D * M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

aoi_mm1 = calculate_deltamm1(tauc_values)
aoi_dm1 = calculate_deltadm1(tauc_values)

# 创建图表和两个子图
fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# 绘制数据
ax_upper.plot(tauc_values, aoi_mm1, label='M/M/1', linestyle='-', color='blue')
ax_upper.set_ylabel(r'$\Delta$')
ax_upper.grid(True)
ax_upper.legend()

ax_lower.plot(tauc_values, aoi_dm1, label='D/M/1', linestyle='--', color='orange')
ax_lower.set_xlabel(r'$\tau_c$')
ax_lower.set_ylabel(r'$\Delta$')
ax_lower.grid(True)
ax_lower.legend()

plt.tight_layout()
plt.savefig('fig/compareTauc_subplots.pdf')
plt.show()
