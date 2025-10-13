import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# 定义常数
mu = 1
M = 3
tau_c = 0.003
rho = 0.53
# p_d = 0.02
D = 0.6

# 生成 rho 的值
pd_values = np.linspace(0.01, 0.61, 100)

def calculate_deltamm1(p_d):
    term1 = 1 / mu
    lambda_values = rho * mu
    term2 = 1 + 1 / rho + (rho**2) / (1 - rho) * np.exp(-lambda_values* M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

def calculate_deltadm1(p_d):
    mu = 1 / (D * rho)
    beta = -rho * lambertw(-rho ** (-1) * np.exp(-1/rho))
    term1 = 1
    # term2 = 1 + 1 / (2 * rho) + beta / (1 - beta)
    term2 = 1 + 1 / (2 * rho) + beta / (1 - beta) * np.exp(-D * M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

aoi_mm1 = calculate_deltamm1(pd_values)
aoi_dm1 = calculate_deltadm1(pd_values)

#绘图
plt.plot(pd_values, aoi_mm1, label='M/M/1', linestyle='-')
plt.plot(pd_values, aoi_dm1, label='D/M/1', linestyle='--')
plt.xlabel(r'$p_d$')
plt.ylabel(r'$\Delta$')
plt.grid(True)
plt.legend()  # 添加图例
plt.savefig('fig/comparePd.pdf')
plt.show()

