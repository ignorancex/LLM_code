from utils.train_utils import *
from config import *
from Net.Net import *
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d = config["d"]
l1, l2 = config["l1"], config["l2"]
m0, n, M, N = config["m0"], config["n"], config["M"], config["N"]
mul_ls = config["mul_ls"]
N_max = N + M
para_ls = config["para_ls"]
m_ls = m0 * mul_ls
Ltype, Vtype = config["Ltype"], config["Vtype"]
### test vary a
data_name = f'Test_data_M_{M}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    test_data_ls = pickle.load(f)

data_name = f'Test_same_data_M_{M}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    test_same_data_ls = pickle.load(f)


### first, we test op model
# model_save_path = f"trained_op_model_N_{N}_n_{n}.pt"
# mdl = torch.load(model_save_path, weights_only=False, map_location=device)

mdl = PQModelop(d).to(device)


l2_ls, h1_ls = [], []

for tmp_data in test_same_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2)
    h1_ls.append(h1)



for tmp_data in test_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2e, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2e)
    h1_ls.append(h1)

l2_vec = np.array(l2_ls)
l2_mat = l2_vec.reshape(len(para_ls)+1, len(mul_ls))

h1_vec = np.array(h1_ls)
h1_mat = h1_vec.reshape(len(para_ls)+1, len(mul_ls))


op_l2_mat = l2_mat


mdl = PQModelbad2(d).to(device)
l2_ls, h1_ls = [], []

for tmp_data in test_same_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2e, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2e)
    h1_ls.append(h1)



for tmp_data in test_data_ls:
    x_test, y_test, A_test, Am_test = tmp_data
    x_test, y_test, A_test, Am_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                                       (x_test, y_test, A_test, Am_test))

    _, l2, h1 = eval_mdl(mdl, A_test, Am_test, x_test, y_test)
    l2_ls.append(l2)
    h1_ls.append(h1)

l2_vec = np.array(l2_ls)
l2_mat = l2_vec.reshape(len(para_ls)+1, len(mul_ls))

h1_vec = np.array(h1_ls)
h1_mat = h1_vec.reshape(len(para_ls)+1, len(mul_ls))

bad_l2_mat = l2_mat

npy_name = 'RM_Result_diversity_test.npy'
with open(npy_name, 'wb') as s:
    np.save(s, m_ls)
    np.save(s, para_ls)
    np.save(s, op_l2_mat)
    np.save(s, bad_l2_mat)

#
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.loglog(m_ls, l2_mat[0, ...], label=r'$V(x)=c \mathbf{I}, c \sim U(1,2)$')
# for i in range(1, len(para_ls)):
#     plt.loglog(m_ls, l2_mat[i, ...], '--', label=rf'$V(x) \sim \mathbf{{U}}({str(para_ls[i][0])},{str(para_ls[i][1])})$')
#
# plt.title(f'd={d},n={n},N={20000}', fontsize=18)
# plt.xlabel(r'$m$', fontsize=18)
# plt.ylabel(r'Relative $L^2$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(fontsize=12)
# plt.savefig(f'div_bad_1_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.png', bbox_inches = 'tight')
# plt.show()