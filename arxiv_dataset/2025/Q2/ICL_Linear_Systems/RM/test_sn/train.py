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
n0, M, N = config["n0"], config["M"], config["N"]
mul_ls = config["mul_ls"]
m_ls = config["m_ls"]
N_max = N + M
Ltype, Vtype = config["Ltype"], config["Vtype"]
n_ls = n0 * mul_ls
n_ls = n_ls.astype(int)

project_name = f'RM_test_n_vary_m_N_{N}_m0_{m_ls[0]}_n0_{n_ls[0]}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}'

### Load data
data_name = f'Train_data_N_{N}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pkl'
with open(data_name, 'rb') as f:
    train_data_ls = pickle.load(f)

### training
for i, n in enumerate(n_ls):
    tmp_data = train_data_ls[i]
    x_train, y_train, A, An = tmp_data
    train_name = f'train_n_{n}'
    if use_wandb:
        wandb.init(project=project_name, name=train_name, reinit=True, config={
            "n": n,
            "learning_rate": train_config["learning_rate"],
            "decay_rate": train_config["decay_rate"],
            "decay_epoch": train_config["decay_epoch"],
            "epochs": train_config["epochs"],
        })

    mdl = train_mdl(d, A, An, x_train, y_train, use_wandb=use_wandb, num_epochs=train_config["epochs"], lr=train_config["learning_rate"],
                    decay_epoch=train_config["decay_epoch"], decay_rate=train_config["decay_rate"])


    model_save_path = f"trained_model_test_n_var_m_N_{N}_n_{n}_l1_{num2str_decimal(l1)}_l2_{num2str_decimal(l2)}_Ltype_{Ltype}_Vtype_{Vtype}.pt"
    torch.save(mdl, model_save_path)

