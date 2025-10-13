from utils.data_utils import *
from config import *
from utils.train_utils import *
import pickle

np.random.seed(925)
torch.manual_seed(925)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

d, l1, l2 = config["d"], config["l1"], config["l2"]
n, M, N0 = config["n"], config["M"], config["N0"]
mul_ls = config["mul_ls"]
m_ls = config["m_ls"]
N_ls = N0 * mul_ls
N_ls = N_ls.astype(int)
N_max = N_ls[-1]
Ltype, Vtype = config["Ltype"], config["Vtype"]

### prepare data
## train and test
x_train = torch.randn(N_max, d, device=device)
A = generate_A_inv(N_max, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype)
y_train = torch.bmm(A, x_train.unsqueeze(2)).squeeze(2)
Yn = generate_Yn_small(N_max, n, d)
An = A @ Yn
A, An, x_train, y_train = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                               (A, An, x_train, y_train))

data_name = f'Train_data_N_{N_max}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'

data = {
    'x_train': x_train,
    'y_train': y_train,
    'A': A,
    'An': An
}

with open(data_name, 'wb') as f:
    pickle.dump(data, f)

## test
x_test = torch.randn(M, d, device=device)
A_test = generate_A_inv(M, d, l1=l1, l2=l2, Ltype=Ltype, Vtype=Vtype)
y_test = torch.bmm(A_test, x_test.unsqueeze(2)).squeeze(2)

A_test, x_test, y_test = (torch.tensor(arr, dtype=torch.float32, device=device) for arr in
                          (A_test, x_test, y_test))

##### prepare test set
test_data_ls = []
for m in m_ls:
    Ym = generate_Yn(M, m, d)
    Am_test = A_test @ Ym
    Am_test = torch.tensor(Am_test, dtype=torch.float32, device=device)
    tmp_data = [x_test, y_test, A_test, Am_test]
    test_data_ls.append(tmp_data)

data_name = f'Test_data_M_{M}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'wb') as f:
    pickle.dump(test_data_ls, f)