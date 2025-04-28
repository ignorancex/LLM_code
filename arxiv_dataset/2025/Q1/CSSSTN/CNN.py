import numpy as np
from sklearn.model_selection import *
from G_D import *
import os
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):  # 支持下标操作 
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

for sub in range(10):
    data_size = 2200
    data = np.load(f'') #[2200, 62, 64, 64]
    label = np.load(f'') # [2200, ]
    
    bac_max = 0
    
    p = 5
    input_shape = (opt.channels, opt.height, opt.width)
    count = 0
    skf = StratifiedKFold(n_splits=5)
    model_acc = list()
    X, Y = data, label
    for train_index, test_index in skf.split(X, Y):
        
        x_train, x_test = X[train_index].astype(np.float32), X[test_index].astype(np.float32)
        y_train, y_test = Y[train_index], Y[test_index]

        x_train = torch.from_numpy(x_train).to(torch.float32)
        x_test = torch.from_numpy(x_test).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(torch.long)
        y_test = torch.from_numpy(y_test).to(torch.long)

        print('the split is:', count)
        print("number of training examples = " + str(x_train.shape[0]))
        print("number of test examples = " + str(x_test.shape[0]))

        D = cnn().cuda()
        
        weights = torch.tensor([1.0, 10.0], dtype=torch.float).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        optimizer = torch.optim.Adam(params=D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )

        trainloader = DataLoader(DiabetesDataset(x_train, y_train), batch_size=opt.batch_size, shuffle=True)

        baclist = list()

        for epoch in range(opt.n_epochs):
            running_loss = 0.0
            c = 0
            correct = 0.0
            total = 0
            for i, data in enumerate(trainloader):
                x, y = data
                x, y = x.cuda(), y.cuda()
                
                optimizer.zero_grad()
                out, _, _, _ = D(x)

                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(out, 1)

                correct += torch.eq(pred, y).sum().float().item()

                total += y.size(0)

                acc_tr = float(correct) / total

                running_loss += loss.item()

                c = i
            print('======>>>>>>[%d] Train Loss: %.3f  Train ACC: %.3f' %
                (epoch + 1, running_loss / c, acc_tr))
            correct = 0
            total = 0
            with torch.no_grad():

                x_test = x_test.cuda()
                out, _, _, _ = D(x_test)
                preds = torch.argmax(out, dim=1).detach().cpu().numpy()
                idsN = np.where(y_test == 0)[0]  # non-target
                idsP = np.where(y_test == 1)[0]  # target
                bac = 0.5 * (np.mean((preds[idsP] == 1)) + np.mean((preds[idsN] == 0)))
                print('Val Bac = {:.5f}'.format(bac))
                
            baclist.append(bac)
            if bac > bac_max:
                # TODO:: 根据被试和训练的数据量保存模型
                torch.save(D.state_dict(),f"")
                print("model has been saved")
                bac_max = bac
        accuracy = max(baclist)
        model_acc.append(accuracy)

    print('sub ', sub)
    model_acc = np.array(model_acc)
    print('model_acc:', model_acc)
    print('min', np.min(model_acc))
    print('max', np.max(model_acc))
    print('mean', np.mean(model_acc))
    print('std', np.std(model_acc))
    print("number:", p)
