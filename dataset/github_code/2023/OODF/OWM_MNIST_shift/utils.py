import torch
import time

def get_optimizer(net, optimizer_type, lr, momentum, model_name):
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    else:
        
        if model_name == 'owm_cnn' : 
            fc1_params = list(map(id, net.fc1.parameters()))
            fc2_params = list(map(id, net.fc2.parameters()))
            fc3_params = list(map(id, net.fc3.parameters()))
            base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params, net.parameters())
            optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': net.fc1.parameters(), 'lr': lr},
                {'params': net.fc2.parameters(), 'lr': lr},
                {'params': net.fc3.parameters(), 'lr': lr}
            ], lr = lr, momentum = 0.9)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = momentum)
    return optimizer

def evaluate_accuracy(eval_set, net, device = None, id = None, stat = False, batch_size = 128):
    data_iter = torch.utils.data.DataLoader(eval_set, batch_size = batch_size, shuffle = False)
    statlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            if net.type == 'fc': X = X.view(X.shape[0], -1).to(device) 
            else: X = X.to(device)
            y = y.to(device)
            if net.type == 'fc': y_hat, y_hidden = net(X)
            else: y_hat, h_list, x_list = net(X)
            if id != None:
                c = torch.ones((1, y_hidden.shape[1]), device = device)
                c.data[0, id : min(id + 600, 799)] = 0
                y_hidden = y_hidden * c
                y_hat = net.layer_2(net.afun_1(y_hidden))
            test_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            for i in range(len(statlist)):
                statlist[i] += (y_hat.argmax(dim=1) == i * torch.ones(y.shape).to(device)).float().sum().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    if stat == False: return test_acc_sum / n  
    else: return test_acc_sum / n, statlist

def get_alpha_array(owm_alpha_list, owm_lambda_list, lamda):
    alpha_array = []
    for i in range(len(owm_alpha_list)):
        if i == len(owm_alpha_list) - 1:
            lamda = 1
        alpha_array.append(owm_alpha_list[i] * owm_lambda_list[i] ** lamda)
    return alpha_array

def pro_weight(p, x, w, alpha = 1.0, cnn = False, stride = 1):
    # print(p.device, x.device)
    if cnn:
        _, _, H, W = x.shape
        F, _, HH, WW = w.shape
        S = stride  # stride
        Ho = int(1 + (H - HH) / S)
        Wo = int(1 + (W - WW) / S)
        for i in range(Ho):
            for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                # r = r[:, range(r.shape[1] - 1, -1, -1)]
                k = torch.mm(p, torch.t(r))
                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
        w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
    else:
        # r = torch.mean(x, 0, True)
        r = x
        k = torch.mm(p, torch.t(r))
        p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
        w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

def train(net, loss, train_set, test_set, optimizer, device, args):
    train_iter = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    net = net.to(device)
    batch_size = args.batch_size

    all_step = args.num_epochs * (len(train_set) // batch_size)
    current_step = 0
        
    for epoch in range(args.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            net.train()
            X = X.view(X.shape[0], -1).to(device) 
            y = y.to(device)
                
            y_hat, hidden = net(X)
            l = loss(y_hat, y) + args.lambda_loss * torch.norm(net.layer_1.W.weight) + args.lambda_loss * torch.norm(net.layer_2.W.weight)

            optimizer.zero_grad()
            l.backward()

            if args.CL_method == 'owm':
                with torch.no_grad():
                    lamda = current_step / all_step
                    alpha_array = get_alpha_array(args.owmfc_alpha_list, args.owmfc_lambda_list, lamda)
                    pro_weight(net.layer_1.P.weight, torch.mean(X, 0, True), net.layer_1.W.weight, alpha_array[0])
                    pro_weight(net.layer_2.P.weight, torch.mean(net.afun_1(hidden), 0, True), net.layer_2.W.weight, alpha_array[1])
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipgrad)

            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            current_step += 1
        # print('Here ! ! ! ! !')
        test_acc = evaluate_accuracy(test_set, net, batch_size = args.batch_size) if test_set != None else 0
        if args.print_frequency != 0 and (epoch + 1) % args.print_frequency == 0:
            print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.2f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
