from config import *
from utils.train_utils import *
import pickle
import wandb

np.random.seed(925)
torch.manual_seed(925)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

use_wandb = False

if use_wandb:
    wandb.login(key='a31c2a5402e98f77d1ca61b14796dfcdeaa2ea5c')

d, l1, l2 = config["d"], config["l1"], config["l2"]
n, M, N0 = config["n"], config["M"], config["N0"]
mul_ls = config["mul_ls"]
m_ls = config["m_ls"]
N_ls = N0 * mul_ls
N_ls = N_ls.astype(int)
N_max = N_ls[-1]
Ltype, Vtype = config["Ltype"], config["Vtype"]

project_name = f'RM_N_vary_m_n_{n}_N0_{N0}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}'

### Load data
data_name = f'Train_data_N_{N_max}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    data = pickle.load(f)

x_train = data['x_train']
y_train = data['y_train']
A = data['A']
An = data['An']
### training
for N in N_ls:
    tmp_A, tmp_An, tmp_x_train, tmp_y_train = A[:N, ...], An[:N, ...], x_train[:N, ...], y_train[:N, ...]
    train_name = f'train_N_{N}'

    if use_wandb:
        wandb.init(project=project_name, name=train_name, reinit=True, config={
            "N": N,
            "learning_rate": train_config["learning_rate"],
            "decay_rate": train_config["decay_rate"],
            "decay_epoch": train_config["decay_epoch"],
            "epochs": train_config["epochs"],
        })

    mdl = train_mdl(d, tmp_A, tmp_An, tmp_x_train, tmp_y_train, init_weight='op', use_wandb=use_wandb, num_epochs=train_config["epochs"], lr=train_config["learning_rate"],
                    decay_epoch=train_config["decay_epoch"], decay_rate=train_config["decay_rate"])


    model_save_path = f"trained_model_test_N_var_m_N_{N}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pt"
    torch.save(mdl, model_save_path)

