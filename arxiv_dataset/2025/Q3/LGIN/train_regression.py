from torch_geometric.datasets import ZINC, AQSOL
from torch_geometric.loader import DataLoader
from model_config import EPOCHS, EPSILON, LR, CLIP_VALUE, EPS, NUM_LAYERS_MLP, C_IN, C_OUT, DROPOUT, USE_ATT, USE_BIAS, TRAINING_CURVATURE, BATCH_SIZE, SAVE_PATH
import torch_geometric.transforms as T
from optimizers.radam import RiemannianAdam
from models.model import MultilayerGINRegression
from torch.nn.functional import l1_loss
import os
from dotenv import load_dotenv
from torch import nn
import torch
import wandb


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    return ave_grads


def train_epoch():
    epoch_loss = 0
    mae = 0

    for step, data in enumerate(train_loader):
        x, adj = data.x.float(), data.edge_index
        x = x.view(x.size(0), 1)
        data.y = data.y.view(data.y.size(0), 1)
        input = (x, adj)

        # Since one dimensional features
        optimizer.zero_grad()
        logits = model(input, batch=data.batch)
        loss = loss_function(logits, data.y.float())

        loss.backward()

        # Gradient clip
        max_norm = CLIP_VALUE
        all_params = list(model.parameters())
        for param in all_params:
            nn.utils.clip_grad_norm_(param, max_norm)

        # Riemennian Optimization
        optimizer.step()

        epoch_loss += loss.item()

        # mae_loss
        mae += l1_loss(logits, data.y.float()).item()

    return epoch_loss/(step+1), mae/(step+1)


@torch.no_grad()
def val_epoch():
    epoch_loss = 0
    mae = 0
    for step, data in enumerate(val_loader):
        x, adj = data.x.float(), data.edge_index
        x = x.view(x.size(0), 1)
        data.y = data.y.view(data.y.size(0), 1)
        input = (x, adj)

        logits = model(input, batch=data.batch)

        loss = loss_function(logits, data.y.float())

        epoch_loss += loss.item()

        mae += l1_loss(logits, data.y.float()).item()

    return epoch_loss/(step+1), mae/(step+1)


@torch.no_grad()
def test():
    epoch_loss = 0
    mae = 0
    for step, data in enumerate(test_loader):
        x, adj = data.x.float(), data.edge_index
        x = x.view(x.size(0), 1)
        data.y = data.y.view(data.y.size(0), 1)
        input = (x, adj)

        logits = model(input, batch=data.batch)

        loss = loss_function(logits, data.y.float())

        epoch_loss += loss.item()

        mae += l1_loss(logits, data.y.float()).item()

    return epoch_loss/(step+1), mae/(step+1)


def training_loop():

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_mae = train_epoch()
        model.eval()

        with torch.no_grad():
            val_loss, val_mae = val_epoch()

            print("Epoch: ", epoch+1)
            print("Train Loss: ", train_loss)
            print("Train MAE: ", train_mae)
            print("Validation Loss: ", val_loss)
            print("validation MAE: ", val_mae)

            grads = plot_grad_flow(model.named_parameters())

            test_loss, test_mae = test()
            print(f"Test Loss: {test_loss}")
            print(f"Test MAE: {test_mae}")

            wandb.log({
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Gradients": wandb.Histogram(grads),
                "Test Loss": test_loss,
                "Train MAE": train_mae,
                "Validation MAE": val_mae,
                "Test MAE": test_mae
            })

            if SAVE_PATH is not None:
                # You may edit the metric for saving here such as test_loss[i]<test_loss[i-1]
                if (epoch+1) % 10 == 0:
                    save_path = SAVE_PATH+f"/Run_1/model_{epoch+1}.pt"
                    # Save weights here
                    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    load_dotenv('.env')
    # torch.autograd.set_detect_anomaly(True)

    inp_name = input('Enter the dataset to be downloaded: ')
    zinc = os.getenv('zinc')
    aqsol = os.getenv('aqsol')

    if inp_name == 'zinc':
        train_set = ZINC(root=zinc, split='train')
        val_set = ZINC(root=zinc, split='val')
        test_set = ZINC(root=zinc, split='test')
    elif inp_name == 'aqsol':
        train_set = AQSOL(root=aqsol, split='train',
                          transform=(T.RemoveIsolatedNodes()))
        val_set = AQSOL(root=aqsol, split='val',
                        transform=(T.RemoveIsolatedNodes()))
        test_set = AQSOL(root=aqsol, split='test',
                         transform=(T.RemoveIsolatedNodes()))

    params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0,
    }

    num_in_features = train_set.num_node_features

    train_loader = DataLoader(train_set, **params)
    val_loader = DataLoader(val_set, **params)
    test_loader = DataLoader(test_set, **params)

    model = MultilayerGINRegression(eps=EPS, num_layers_mlp=NUM_LAYERS_MLP, c_in=C_IN,
                                    c_out=C_OUT, in_features=num_in_features, dropout=DROPOUT, use_att=USE_ATT, use_bias=USE_BIAS, training=TRAINING_CURVATURE)
    optimizer = RiemannianAdam(
        params=model.parameters(), lr=LR, weight_decay=EPSILON)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=50)
    loss_function = nn.L1Loss()

    wandb.init(
        project="Lorentzian Graph Isomorphism Network"
    )

    training_loop()
