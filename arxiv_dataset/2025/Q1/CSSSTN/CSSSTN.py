import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import os
from types import SimpleNamespace
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DiabetesDataset(Dataset):
    def __init__(self, data1, data2, label):
        self.data1 = data1
        self.data2 = data2
        self.label = label

    def __getitem__(self, index):  # 支持下标操作
        return self.data1[index], self.data2[index], self.label[index]

    def __len__(self):
        return len(self.label)

opt = {
    "epoch": 0,  # epoch to start training from
    "n_epochs": 50,  # number of epochs of training
    "batch_size": 10,  # size of the batches
    "lr": 1e-3,  # adam: learning rate
    "b1": 0.5,  # adam: decay of first order momentum of gradient
    "b2": 0.999,  # adam: decay of first order momentum of gradient
    "decay_epoch": 30,  # epoch from which to start lr decay
}
opt = SimpleNamespace(**opt)
print(opt)
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


data_size = 2200

from G_D import *

# target subject list
sub1s = []
# source subject list
sub2s = []

for sub1 in sub1s:
    for sub2 in sub2s:
        golden_data = np.load(f'')
        golden_lable = np.load(f'')

        transferred_data = np.load(f'')
        transferred_label = np.load(f'')

        X, Y = transferred_data, transferred_label
        acc_max = 0

        p = 5
        count = 0
        skf = StratifiedKFold(n_splits=p)
        model_acc = list()

        for train_index, test_index in skf.split(X, Y):
            count += 1
            x_train, golden_train, x_test = X[train_index].astype(np.float32), golden_data[train_index].astype(
                np.float32), X[test_index].astype(np.float32)
            golden_train_lable = golden_lable[train_index]
            y_train, y_test = Y[train_index], Y[test_index]

            golden_class_0 = golden_train[np.where(golden_train_lable == 0)]
            golden_class_1 = golden_train[np.where(golden_train_lable == 1)]

            # target data index
            one_indices = np.where(y_train == 1)[0]
            np.random.shuffle(one_indices)
            replicated_indices = np.repeat(one_indices, 10)

            expanded_transferred_data = x_train[replicated_indices]  
            expanded_label = np.ones(len(expanded_transferred_data))

            x_train = np.concatenate([x_train, expanded_transferred_data], axis=0)
            y_train = np.concatenate([y_train, expanded_label], axis=0)
            expanded_golden_data = np.repeat(golden_train[one_indices], 4, axis=0)

            golden_train = np.concatenate([golden_train, expanded_golden_data], axis=0)
            golden_train_lable = np.concatenate([golden_train_lable, np.ones(len(expanded_golden_data))], axis=0)

            x_train = torch.from_numpy(x_train).to(torch.float32)

            golden_class_0 = torch.from_numpy(golden_class_0).to(torch.float32).cuda()
            golden_class_1 = torch.from_numpy(golden_class_1).to(torch.float32).cuda()

            x_test = torch.from_numpy(x_test).to(torch.float32)
            y_train = torch.from_numpy(y_train).to(torch.long)
            y_test = torch.from_numpy(y_test).to(torch.long)
            print('the split is:', count)
            print("number of training examples = " + str(x_train.shape[0]))
            print("number of golden training examples = " + str(golden_train.shape[0]))
            print("number of test examples = " + str(x_test.shape[0]))
            del expanded_golden_data, expanded_transferred_data, expanded_label,one_indices,golden_train_lable

            Loss_Style = torch.nn.KLDivLoss(reduction='batchmean')

            Loss_Cont = torch.nn.MSELoss()

            criterion = nn.CrossEntropyLoss()

            G_AB = Generator()
            G_AB = G_AB.cuda()
            DA = cnn()  # target classification
            DB = cnn()  # source classification

            DA.load_state_dict(torch.load(f""))
            DB.load_state_dict(torch.load(f""))

            DA = DA.cuda()
            DB = DB.cuda()

            for name, param in DA.named_parameters():
                param.requires_grad = False
            for name, param in DB.named_parameters():
                param.requires_grad = False

            optimizer_G = torch.optim.Adam(params=G_AB.parameters(),
                                        lr=opt.lr, betas=(opt.b1, opt.b2))

            lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
                optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )

            trainloader = DataLoader(
                DiabetesDataset(x_train, golden_train, y_train),
                batch_size=opt.batch_size,
                shuffle=True
            )

            baclist = list()

            # get class template
            golden_class_0_mean = golden_class_0.mean(dim=0)
            golden_class_1_mean = golden_class_1.mean(dim=0)
            for epoch in range(opt.n_epochs):
                G_AB.train()
                running_loss = 0.0
                c = 0
                correct = 0.0
                total = 0

                for i, data in enumerate(trainloader):
                    A, _, C = data  # A: 弱被试特征, C: 类别标签
                    A, C = A.cuda(), C.cuda()

                    optimizer_G.zero_grad()

                    fake_A = G_AB(A)

                    # Select the corresponding category template according to the category
                    golden_target = torch.stack([
                        golden_class_0_mean if label == 0 else golden_class_1_mean
                        for label in C
                    ]).cuda()

                    # Using source subject classifiers to obtain high-level features
                    D_fake_A_logits, _, _, D_fake_A_high = DB(fake_A)
                    _, _, _, D_golden_high = DB(golden_target)

                    # Use target subject classifier to obtain content consistency
                    _, _, _, D_real_A_high = DA(A)

                    # Style loss: Generate features close to the source subject of the corresponding category
                    style_loss = Loss_Style(D_fake_A_high, D_golden_high)

                    # Content loss: keep the content consistency between the generated features and the original features
                    content_loss = Loss_Cont(D_fake_A_high, D_real_A_high)

                    # Classification loss: The classification results of the generated features are consistent with the true labels
                    classification_loss = criterion(D_fake_A_logits, C)

                    # all loss
                    loss = style_loss + content_loss + classification_loss

                    loss.backward()
                    optimizer_G.step()

                    pred = torch.argmax(D_fake_A_logits, 1)

                    correct += torch.eq(pred, C).sum().float().item()

                    total += C.size(0)

                    acc_tr = float(correct) / total

                    running_loss += loss.item()

                    c = i
                print('======>>>>>>[%d] Train Loss: %.3f  Train ACC: %.3f' %
                    (epoch + 1, running_loss / c, acc_tr))

                correct = 0
                total = 0
                # test batch size
                batch_size = 10
                G_AB.eval()
                with torch.no_grad():
                    res = []
                    num_batches = (x_test.shape[0] + batch_size - 1) // batch_size

                    for i in range(num_batches):
                        # Splitting test data
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, x_test.shape[0])
                        X_test = x_test[start_idx:end_idx].cuda()
                        new_X_test = G_AB(X_test)

                        out1, _, _, _ = DA(X_test)
                        out2, _, _, _ = DB(new_X_test)

                        res.append(out1 + out2)

                    res = torch.cat(res, dim=0)
                    preds = torch.argmax(res, dim=1).detach().cpu().numpy()

                    idsN = np.where(y_test == 0)[0]  # non-target
                    idsP = np.where(y_test == 1)[0]  # target
                    bac = 0.5 * (np.mean(preds[idsP] == 1) + np.mean(preds[idsN] == 0))
                    print(f"{sub1} -> {sub2} Val Bac = {bac:.5f}")

                baclist.append(bac)
                if bac > acc_max:
                    torch.save(G_AB.state_dict(), f"")
                    print("model has been saved")
                    acc_max = bac

            accuracy = max(baclist)
            model_acc.append(accuracy)
            torch.cuda.empty_cache()  # 清理显存

        model_acc = np.array(model_acc)
        print(str(sub1) + " -> " + str(sub2))
        print('model_acc:', model_acc)
        print('min', np.min(model_acc))
        print('max', np.max(model_acc))
        print('mean', np.mean(model_acc))
        print('std', np.std(model_acc))
        print("number:", p)
