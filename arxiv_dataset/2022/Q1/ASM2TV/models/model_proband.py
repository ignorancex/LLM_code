from cca_grad import CCALoss
from data_proband import MyDataset

from sklearn import metrics
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import torchvision.models as Models
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


class SharedFeatureLayer(nn.Module):
    def __init__(self, input, kind='fnn'):
        super(SharedFeatureLayer, self).__init__()
        if kind == 'fnn':
            self.feature = nn.Sequential(
                nn.Linear(input, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(),
                nn.ReLU(inplace=True),
            )
        else:
            self.feature = Models.resnet18(pretrained=True)
            self.feature = torch.nn.Sequential(*list(self.feature.children())[:-1])

    def forward(self, x):
        x = self.feature(x)
        return x


class SpecificFeatureLayer(nn.Module):
    def __init__(self, input):
        super(SpecificFeatureLayer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.embedding(x)
        return x


class TaskLayer(nn.Module):
    def __init__(self, c):
        super(TaskLayer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Softplus(),

            nn.Linear(256, c),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def my_split(tensor, split_size, dim=1):
    dim_size = tensor.size(dim)
    assert np.sum(np.array(split_size)) == dim_size
    num_splits = len(split_size)

    def get_split_size(i):
        s = 0
        for j in range(i):
            s += split_size[j]
        return s
    return tuple(tensor.narrow(dim, get_split_size(i), split_size[i]) for i
                 in range(0, num_splits))


def main(log_file, gpu_id, training=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 读取log文件
    log = []
    with open(log_file, "r") as f:
        for row in f:
            row = row.strip()
            if len(row) > 0:
                log.append(row)
    task_num, view_num = log[0].split()
    task_num = int(task_num)
    view_num = int(view_num)
    view_dim = [0 for i in range(view_num)]
    tasks = {}
    for i in range(task_num):
        tasks[i] = [[], []]
        _, _, c, v = log[3 * i + 1].split()
        c = int(c)
        v = int(v)
        lst = log[3 * i + 3].split()
        for j in range(v):
            tasks[i][0].append(int(lst[2 * j]))
            tasks[i][1].append(int(lst[2 * j + 1]))
            view_dim[int(lst[2 * j])] = int(lst[2 * j + 1])

    # 构造模型
    # share layer
    share_models = []
    for i in range(view_num):
        share_models.append(SharedFeatureLayer(view_dim[i]))

    # task layer
    task_models = []
    for i in range(task_num):
        task_models.append(TaskLayer(c))

    # view-task specufic
    view_task = [[None for i in range(task_num)] for j in range(view_num)]
    for i in range(view_num):
        for j in range(task_num):
            if i in tasks[j][0]:
                view_task[i][j] = SpecificFeatureLayer(1024)

    if training is True:
        print("start training | gpu id: ", gpu_id)
        train_data = [MyDataset(i, train=True) for i in range(task_num)]
        train_loader = [Data.DataLoader(dataset=train_data[i], batch_size=512, shuffle=True, num_workers=12)
                        for i in range(task_num)]

        for i in range(task_num):
            print("task ", i, " | train data size: ", len(train_data[i]))

            for iter in range(1):
               '''for i in range(task_num):
                    torch.save(task_models[i].state_dict(), './proband/params_20news_task_' + str(i) + '.pkl')
                for i in range(view_num):
                    torch.save(share_models[i].state_dict(), './proband/params_20news_share_' + str(i) + '.pkl')
                for i in range(view_num):
                    for j in range(task_num):
                        if view_task[i][j] is not None:
                            torch.save(view_task[i][j].state_dict(),
                                       './proband/params_20news_view_' + str(i) + '_task_'
                                       + str(j) + '.pkl')

        for i in range(task_num):
            task_models[i].load_state_dict(torch.load('./proband/params_20news_task_' + str(i) + '.pkl'))

        for i in range(view_num):
            share_models[i].load_state_dict(torch.load('./proband/params_20news_share_' + str(i) + '.pkl'))

        for i in range(view_num):
            for j in range(task_num):
                if view_task[i][j] is not None:
                    view_task[i][j].load_state_dict(torch.load('./proband/params_20news_view_' + str(i) + '_task_'
                                                            + str(j) + '.pkl'))'''
            # Task训练
            # 初始化相关性矩阵
            M = torch.rand((task_num, task_num))
            M = M / torch.trace(M)
            lamda = 1e-6
            print(lamda)
            task_parameter = []
            for i in range(view_num):
                share_models[i].cpu()
                share_models[i].train()
                task_parameter.append({'params': share_models[i].parameters()})
                for j in range(task_num):
                    if i in tasks[j][0]:
                        view_task[i][j].train()
                        view_task[i][j].cpu()
                        task_parameter.append({'params': view_task[i][j].parameters()})
            for i in range(task_num):
                task_models[i].cpu()
                task_models[i].train()
                task_parameter.append({'params': task_models[i].parameters()})
            optimizer = optim.Adam(task_parameter, lr=1e-3)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            for epoch in range(10):
                scheduler.step()
                m = Variable(torch.inverse(M), requires_grad=False).cpu()
                for i in range(task_num):
                    task_models[i].train()
                for i in range(view_num):
                    share_models[i].train()
                    for j in range(task_num):
                        if view_task[i][j] is not None:
                            view_task[i][j].train()
                for i in range(task_num):
                    train_loss = 0
                    for step, (x, y) in enumerate(train_loader[i]):
                        x = torch.squeeze(x)
                        b_y = Variable(torch.LongTensor(y)).cpu()
                        xs = my_split(x, tasks[i][1])

                        # forward
                        z = []
                        for j in range(len(tasks[i][0])):
                            k = tasks[i][0][j]
                            b_x = Variable(xs[j]).cpu()
                            z.append(task_models[i](view_task[k][i](share_models[k](b_x))))

                        z = torch.stack(z)
                        z = torch.mean(z, 0)

                        loss_func = nn.CrossEntropyLoss()
                        loss = loss_func(z, b_y)

                        # 正则项
                        for j in range(task_num):
                            wi = nn.Softplus()(task_models[i].classifier[0].weight)
                            wj = nn.Softplus()(task_models[j].classifier[0].weight)
                            loss += lamda * m[i, j] * ((torch.sum(torch.mul(wi, wj))) + torch.sum(torch.mul(
                                task_models[i].classifier[1].weight, task_models[j].classifier[1].weight)))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * x.size(0)
                    print('Epoch ' + str(epoch) + ' | task ' + str(i) + ' | train loss: %.9f' % (train_loss / len(train_data[i])))

                for i in range(task_num):
                    task_models[i].eval()
                for i in range(view_num):
                    share_models[i].eval()
                    for j in range(task_num):
                        if view_task[i][j] is not None:
                            view_task[i][j].eval()

                for i in range(task_num):
                    test_data = MyDataset(task_id=i, train=False)
                    test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=6)
                    correct = 0
                    for step, (x, y) in enumerate(test_loader):
                        x = torch.squeeze(x)
                        b_y = torch.LongTensor(y)
                        # split tensor
                        xs = my_split(x, tasks[i][1])

                        # forward
                        z = []
                        for j in range(len(tasks[i][0])):
                            k = tasks[i][0][j]
                            b_x = Variable(xs[j]).cpu()
                            z.append(task_models[i](view_task[k][i](share_models[k](b_x))))
                        z = torch.stack(z)
                        z = torch.mean(z, 0)
                        _, h = torch.max(z, 1)

                        correct += torch.sum(h.cpu().data == b_y)
                    print('Epoch ' + str(epoch) + ' | task ' + str(i) + " |accuracy: ", torch.true_divide(correct, len(test_data)))

                if (epoch + 1) % 2 == 0:
                    # update M
                    w = torch.zeros((task_num, task_num))
                    for i in range(task_num):
                        for j in range(task_num):
                            wi = nn.Softplus()(task_models[i].classifier[0].weight)
                            wj = nn.Softplus()(task_models[j].classifier[0].weight)
                            wij = torch.sum(torch.mul(wi, wj)) + torch.sum(torch.mul(task_models[i].classifier[1].weight,
                                                                                         task_models[j].classifier[1].weight))
                            w[i, j] = wij.cpu().item()

                    e, v = torch.symeig(w, eigenvectors=True)
                    e[e < 0] = 0
                    e = torch.sqrt(e)
                    e = torch.FloatTensor(np.diag(e.numpy()))
                    w = torch.matmul(torch.matmul(v, e), v.t())
                    M = w / torch.trace(w)
    else:
        # 加载模型
        for i in range(task_num):
            task_models[i].load_state_dict(torch.load('./proband/params_20news_task_' + str(i) + '.pkl'))

        for i in range(view_num):
            share_models[i].load_state_dict(torch.load('./proband/params_20news_share_' + str(i) + '.pkl'))

        for i in range(view_num):
            for j in range(task_num):
                if view_task[i][j] is not None:
                    view_task[i][j].load_state_dict(torch.load('./proband/params_20news_view_' + str(i) + '_task_'
                                                               + str(j) + '.pkl'))

    # 测试
    for i in range(task_num):
        task_models[i].eval()
        task_models[i].cpu()
    for i in range(view_num):
        share_models[i].eval()
        share_models[i].cpu()
        for j in range(task_num):
            if view_task[i][j] is not None:
                view_task[i][j].eval()
                view_task[i][j].cpu()

    for i in range(task_num):
        test_result = []
        test_label = []
        test_data = MyDataset(task_id=i, train=False)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=6)
        print("task ", i, " | test data size: ", len(test_data))
        correct = 0
        for step, (x, y) in enumerate(test_loader):
            x = torch.squeeze(x)
            b_y = torch.LongTensor(y)
            # split tensor
            xs = my_split(x, tasks[i][1])

            # forward
            z = []
            for j in range(len(tasks[i][0])):
                k = tasks[i][0][j]
                b_x = Variable(xs[j]).cpu()
                z.append(task_models[i](view_task[k][i](share_models[k](b_x))))
            z = torch.stack(z)
            z = torch.mean(z, 0)
            _, h = torch.max(z, 1)

            correct += torch.sum(h.cpu().data == b_y)
            test_result.append(h.cpu().data)
            test_label.append(b_y)
        test_result = torch.cat(test_result, 0)
        test_label = torch.cat(test_label, 0)

        print('task ' + str(i) + " | accuracy: ", torch.true_divide(correct, len(test_data)))
        test_result = test_result.numpy()
        test_label = test_label.numpy()
        print(metrics.classification_report(test_label, test_result, target_names=[str(i) for i in range(8)], digits=4))
        # 保存
        # torch.save(test_result, "./proband/prediction_20news.pkl", test_result)
        # torch.save(test_label, "./proband/prediction_20news_label.pkl", test_label)

if __name__ == '__main__':
   main('./proband/1/data.log', gpu_id=10)