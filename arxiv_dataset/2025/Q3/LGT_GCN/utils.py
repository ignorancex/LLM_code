import torch
import torch.nn.functional as F

def train(net, modelName,optimizer, criterion, data, depth):
    net.train()
    optimizer.zero_grad()
    output,loss_CI,_,_,hidden_layer = net(data.x, data.adj, depth=depth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_CI = torch.tensor(loss_CI, device=device)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss += loss_CI
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc, 0, 0

def val_and_test(net, modelName, criterion, data, nlayer, depth, epoch):
    net.eval()
    output,loss_CI,metric1,metric2,hidden_layer = net(data.x, data.adj, depth)
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])

    mad = calculate_MAD(hidden_layer[data.test_mask])

    return acc_test,acc_val,loss_val,loss_test,mad,metric1,metric2

def accuracy(output, labels):
    preds = output.max(1)[1]
    preds=preds.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def calculate_MAD(x):  #平均距离测量
    x_norm=F.normalize(x)
    mad = 1-torch.mm(x_norm,x_norm.t())
    return mad.mean()
