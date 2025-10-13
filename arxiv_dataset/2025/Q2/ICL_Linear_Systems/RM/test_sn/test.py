from utils.train_utils import *
from config import *
import wandb
import pickle

np.random.seed(925)
torch.manual_seed(925)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d, l1, l2 = config["d"], config["l1"], config["l2"]
n0, M, N = config["n0"], config["M"], config["N"]
mul_ls = config["mul_ls"]
m_ls = config["m_ls"]
N_max = N + M
Ltype, Vtype = config["Ltype"], config["Vtype"]
n_ls = n0 * mul_ls
n_ls = n_ls.astype(int)

data_name = f'Test_data_M_{M}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    test_data_ls = pickle.load(f)

mse_ls = []

for n in n_ls:
    model_save_path = f"trained_model_test_n_var_m_N_{N}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pt"
    mdl = torch.load(model_save_path, weights_only=False, map_location=device)

    for tmp_data in test_data_ls:
        x_test, y_test, A_test, Am_test = tmp_data
        x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                           (x_test, y_test, A_test, Am_test))

        mse, l2e, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)

        mse_ls.append(mse)

mse_vec = np.array(mse_ls)
mse_mat = mse_vec.reshape(len(n_ls), len(m_ls)).T

ref_N = n_ls[0]
ref_y = mse_mat[0, 0] - mse_mat[0, -1]
scale_2 = ref_y / (ref_N ** (-2.0))

n_ls = n_ls[:-4]
ref_rate = 3e-1 * scale_2 * n_ls**(-2.0)
mse_mat = mse_mat[:, :-4] - mse_mat[:, [-1]]


npy_name = 'RM_Result_test_sn.npy'
with open(npy_name, 'wb') as s:
    np.save(s, n_ls)
    np.save(s, m_ls)
    np.save(s, mse_mat)
    np.save(s, ref_rate)


#mse_mat = mse_vec.reshape(len(m_ls), len(n_ls)).T
# print(mse_mat)
# import matplotlib.pyplot as plt
# plt.figure(1)
# mse_mat = mse_mat[:, 1:]
# n_ls = n_ls[1:]
# # print(mse_mat.shape)
# # zxc
#
# for i in range(len(m_ls)):
#     plt.loglog(n_ls[:-4], mse_mat[i, :-4] - mse_mat[i, -1], '--*', label=f'm={m_ls[i]}')
#
# ref_N = n_ls[0]
# ref_y = mse_mat[0, 0] - mse_mat[0, -1]
# scale_2 = ref_y / (ref_N ** (-2.0))
# scale_1 = ref_y / (ref_N ** (-1.0))
#
# plt.loglog(n_ls[:-4], 1e-1 * scale_2 * n_ls[:-4]**(-2.0), 'b', label=r'$O(n^{-2})$')
# #plt.loglog(n_ls[:-2], scale_2 * n_ls[:-2]**(-2.0), 'b', label=r'$O(n^{-2})$')
#
# #plt.title(f'd={d},N={N}', fontsize=18)
# plt.xlabel(r'$n$', fontsize=18)
# plt.ylabel(r'Shifted $MSE$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12, loc='upper right')
# plt.savefig(f'test_sn_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.png', bbox_inches = 'tight')
# plt.show()