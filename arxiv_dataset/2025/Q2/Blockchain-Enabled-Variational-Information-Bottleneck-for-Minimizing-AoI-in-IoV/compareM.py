import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# 定义常数
mu = 1
rho = 0.53
tau_c = 0.003
p_d = 0.02
D = 0.6

# 生成 rho 的值
M_values = np.linspace(2, 10)

def calculate_deltamm1(M):
    term1 = 1 / mu
    lambda_values = rho * mu
    term2 = 1 + 1 / rho + (rho**2) / (1 - rho) * np.exp(lambda_values* M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

def calculate_deltadm1(M):
    mu = 1 / (D * rho)
    beta = -rho * lambertw(-rho ** (-1) * np.exp(-1/rho))
    term1 = 1
    term2 = 1 + 1 / (2 * rho) + beta / (1 - beta) * np.exp(D * M * (M - 1) * tau_c / 2) / (1 - p_d)
    return term1 * term2

aoi_mm1 = calculate_deltamm1(M_values)
aoi_dm1 = calculate_deltadm1(M_values)

# 创建两个子图
fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# 绘制数据
# ax_upper.plot(M_values, aoi_mm1, label='M/M/1', linestyle='-')
ax_upper.plot(M_values, aoi_dm1, label='D/M/1', linestyle='--',color='orange')
ax_upper.set_ylim(2.2, 2.4)
ax_upper.grid(True)
ax_upper.legend()

ax_lower.plot(M_values, aoi_mm1, label='M/M/1', linestyle='-')
# ax_lower.plot(M_values, aoi_dm1, label='D/M/1', linestyle='--')
ax_lower.set_ylim(3.4, 3.6)
ax_lower.set_xlabel(r'$M$')
ax_lower.set_ylabel(r'$\Delta$')
ax_lower.grid(True)
ax_lower.legend()

# 调整两个子图之间的间距
plt.subplots_adjust(hspace=0.3)

plt.tight_layout()
plt.savefig('fig/compareM.pdf')
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import lambertw
#
# # 定义常数
# mu = 1
# rho = 0.53
# tau_c = 0.003
# p_d = 0.02
# D = 0.6
#
# # 生成 rho 的值
# M_values = np.linspace(2, 10)
#
# def calculate_deltamm1(M):
#     term1 = 1 / mu
#     lambda_values = rho * mu
#     term2 = 1 + 1 / rho + (rho**2) / (1 - rho) * np.exp(lambda_values* M * (M - 1) * tau_c / 2) / (1 - p_d)
#     return term1 * term2
#
# def calculate_deltadm1(M):
#     mu = 1 / (D * rho)
#     beta = -rho * lambertw(-rho ** (-1) * np.exp(-1/rho))
#     term1 = 1
#     term2 = 1 + 1 / (2 * rho) + beta / (1 - beta) * np.exp(D * M * (M - 1) * tau_c / 2) / (1 - p_d)
#     return term1 * term2
#
# aoi_mm1 = calculate_deltamm1(M_values)
# aoi_dm1 = calculate_deltadm1(M_values)
#
# # 创建图表和两个子图
# fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
#
# # 绘制数据
# # ax_lower.plot(M_values, aoi_mm1, label='M/M/1', linestyle='-')
# ax_lower.plot(M_values, aoi_dm1, label='D/M/1', linestyle='--', color = 'orange')
# ax_lower.set_ylim(2.25, 2.35)
# ax_lower.grid(True)
# ax_lower.set_xlabel(r'$M$')
# ax_lower.set_ylabel(r'$\Delta$')
# ax_lower.legend()
#
# ax_upper.plot(M_values, aoi_mm1, label='M/M/1', linestyle='-', color = 'blue')
# # ax_upper.plot(M_values, aoi_dm1, label='D/M/1', linestyle='--')
# ax_upper.set_ylim(3.45, 3.55)
# # ax_upper.set_xlabel(r'$M$')
# ax_upper.set_ylabel(r'$\Delta$')
# ax_upper.grid(True)
# ax_upper.legend()
# ax_upper.legend(loc='lower right')  # 将图例移到右下角
#
# # 隐藏上图的x轴实线
# ax_upper.spines['bottom'].set_visible(False)
#
# # 隐藏下图的最上边实线
# ax_lower.spines['top'].set_visible(False)
#
#
# # 添加断轴线
# d = .015  # 断轴线长度
# kwargs = dict(color='k', clip_on=False)  # 断轴线参数
# ax_upper.plot((1-d, 1+d), (2.4-d, 2.4+d), transform=ax_upper.transAxes, **kwargs)  # 左上角斜线
# ax_lower.plot((1-d, 1+d), (3.4-d, 3.4+d), transform=ax_lower.transAxes, **kwargs)  # 左上角斜线
#
# plt.tight_layout()
# plt.savefig('fig/compareM_with_break.png')
# plt.show()
