from Net.Net import *
from utils.utils import *
from tqdm import tqdm
import torch.optim as optim
import wandb

def my_h1_loss(pd, ref, d):
    k = torch.arange(1, d + 1).to(device)
    loss = torch.mean(torch.sum(((k * np.pi) ** 2) * (pd - ref) ** 2, dim=1))
    return loss


def train_mdl(d, A, An, x_train, y_train, loss_type='l2', init_weight='op', use_wandb = False, num_epochs=2000, lr=0.01, decay_epoch=200, decay_rate=0.95):
    if init_weight == 'op':
        model = PQModelop(d).to(device)
    elif init_weight == 'bad':
        model = PQModelbad(d).to(device)
    elif init_weight == 'far':
        model = PQModel_far(d).to(device)
    elif init_weight == 'bad2':
        model = PQModelbad2(d).to(device)
    else:
        model = PQModel(d).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_rate)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        y_pred = model(An, x_train)
        if loss_type == 'l2':
            loss = nn.MSELoss()(y_pred, y_train)
        else:
            loss = my_h1_loss(y_pred, y_train, d)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        ### log matrix error
        if use_wandb:
            matrix_error = torch.norm(model.P @ A @ model.Q - A, p=2, dim=(1, 2)).mean().item()
            matrix_error_n = torch.norm(model.P @ An @ model.Q - A, p=2, dim=(1, 2)).mean().item()
            relative_matrix_error_n = (torch.norm(model.P @ An @ model.Q - A, p=2, dim=(1, 2))/torch.norm(A, p=2, dim=(1, 2))).mean().item()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": current_lr,
                "Train Matrix error |PA_nQ-A|": matrix_error_n,
                "Train Matrix error |PAQ-A|": matrix_error,
                "Relative Train Matrix error |PAQ-A|": relative_matrix_error_n,
            })

    return model


def eval_mdl(model, A_test, Am_test, x_test, y_test):
    model.eval()
    y_pred_m = model(Am_test, x_test)

    mse_m = nn.MSELoss()(y_pred_m, y_test).item()

    y_pred_m = y_pred_m.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    l2, h1 = get_average_relative_error(y_pred_m, y_test)
    #l2, h1 = get_average_error(y_pred_m, y_test)

    return mse_m, l2, h1