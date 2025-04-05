"""
https://github.com/guyera/Generalized-ODIN-Implementation/blob/master/code/cal.py
"""
import os
import numpy as np
import torch
from torch import optim
import joblib
from tqdm import tqdm

from lolip.models.detectors.deconfnet import DeconfNet, InnerDeconf, CosineDeconf, EuclideanDeconf
from lolip.variables import auto_var
import lolip.models.detectors.deconfnet_models as archs

h_dict = {
    'cosine': CosineDeconf,
    'inner': InnerDeconf,
    'euclid': EuclideanDeconf
}

losses_dict = {
    'ce': torch.nn.CrossEntropyLoss(),
}

def train_model(ds_name, architecture, weight_decay=0.0001, epochs=300, batch_size=64, device="cuda",
              loss_type="ce", similarity="cosine", model_dir="./models/genodin"):
    trnX, trny, _, _, rest = auto_var.get_var_with_argument("dataset", ds_name)
    oodX = np.concatenate((rest[0], rest[1]), axis=0)
    dset = torch.utils.data.TensorDataset(
        torch.from_numpy(trnX.transpose(0, 3, 1, 2)).float(),
        torch.from_numpy(trny).long()
    )
    train_data = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

    n_classes = len(np.unique(trny))
    underlying_net = getattr(archs, architecture)(n_classes=n_classes)

    h = h_dict[similarity](underlying_net.output_size, n_classes)
    h.to(device)

    deconf_net = DeconfNet(underlying_net, underlying_net.output_size, n_classes, h, False)

    deconf_net.to(device)

    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == 'h.h.weight' or name == 'h.h.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr = 0.1, momentum = 0.9, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)

    h_optimizer = optim.SGD(h_parameters, lr = 0.1, momentum = 0.9) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)

    epoch_start = 0
    epoch_loss = None

    #get outlier data
    #train_data, val_data, test_data, open_data = get_datasets(data_dir, data_name, batch_size)

    criterion = losses_dict[loss_type]

    checkpoint_path = f'{model_dir}/{ds_name}_checkpoint.pth'
    model_path = f'{model_dir}/{ds_name}_model.pth'
    # Train the model
    if not os.path.exists(checkpoint_path) and not os.path.exists(model_path):
        deconf_net.train()

        num_batches = len(train_data)
        epoch_bar = tqdm(total = num_batches * epochs, initial = num_batches * epoch_start)

        for epoch in range(epoch_start, epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_data):
                if epoch_loss is None:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{num_batches}')
                else:
                    epoch_bar.set_description(f'Training | Epoch {epoch + 1}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {batch_idx + 1}/{num_batches}')
                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()

                logits, _, _ = deconf_net(inputs)
                loss = criterion(logits, targets)
                loss.backward()

                optimizer.step()
                h_optimizer.step()
                total_loss += loss.item()

                epoch_bar.update()

            epoch_loss = total_loss
            h_scheduler.step()
            scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'h_optimizer': h_optimizer.state_dict(),
                'deconf_net': deconf_net.state_dict(),
                'scheduler': scheduler.state_dict(),
                'h_scheduler': h_scheduler.state_dict(),
                'epoch_loss': epoch_loss,
            }
            torch.save(checkpoint, checkpoint_path) # For continuing training or inference
            torch.save(deconf_net.state_dict(), model_path) # For exporting / sharing / inference only

        if epoch_loss is None:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Batch {num_batches}/{num_batches}')
        else:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {num_batches}/{num_batches}')
        epoch_bar.close()
    else:
        deconf_net.load_state_dict(torch.load(model_path))

    #if test:
    #    deconf_net.eval()
    #    best_val_score = None
    #    best_auc = None
    #    noise_magnitudes = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    #
    #    for score_func in ['h', 'g', 'logit']:
    #        print(f'Score function: {score_func}')
    #        for noise_magnitude in noise_magnitudes:
    #            print(f'Noise magnitude {noise_magnitude:.5f}         ')
    #            validation_results =  np.average(testData(deconf_net, device, val_data, noise_magnitude, criterion, score_func, title = 'Validating'))
    #            print('ID Validation Score:',validation_results)
    #
    #            id_test_results = testData(deconf_net, device, test_data, noise_magnitude, criterion, score_func, title = 'Testing ID') 
    #
    #            ood_test_results = testData(deconf_net, device, open_data, noise_magnitude, criterion, score_func, title = 'Testing OOD')
    #            auroc = calc_auroc(id_test_results, ood_test_results)*100
    #            tnrATtpr95 = calc_tnr(id_test_results, ood_test_results)
    #            print('AUROC:', auroc, 'TNR@TPR95:', tnrATtpr95)
    #            if best_auc is None:
    #                best_auc = auroc
    #            else:
    #                best_auc = max(best_auc, auroc)
    #            if best_val_score is None or validation_results > best_val_score:
    #                best_val_score = validation_results
    #                best_val_auc = auroc
    #                best_tnr = tnrATtpr95
    #
    #    print('supposedly best auc: ', best_val_auc, ' and tnr@tpr95 ', best_tnr)
    #    print('true best auc:'      , best_auc)


def main():
    #ds_name = "mnistwo9"
    #architecture = "CNN002"
    #ds_name = "cifar10wo0"
    ds_name = "cifar100coarsewo0"
    architecture = "WRN_40_10"
    train_model(ds_name, architecture)

if __name__ == "__main__":
    main()
