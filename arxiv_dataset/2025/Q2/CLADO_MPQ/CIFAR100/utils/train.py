import torch,tensorboardX,time
from utils.util import evaluate,generate_variation

def QAVAT_train(model,train_loader,test_loader,config,imgSize=32,imgFlat=False,
                lossfunc=torch.nn.CrossEntropyLoss(),printPerEpoch=100):

    tb = tensorboardX.SummaryWriter(comment=config['trial_name'])
    C = config
    if C['optimizer'] == 'SGD' or C['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=C['lr'],momentum=0.9)
    elif C['optimizer'] == 'adam' or C['optimizer'] == 'Adam' or C['optimizer'] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),lr=C['lr'])
    else:
        print('unrecognized optimizer defined in config')
        exit(0)
    for epoch in range(C['epochs']):
        # lr decay
        current_lr = C['lr'] * (C['decay_ratio'] ** (epoch // C['decay_ep']))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        start = time.time()
        total_loss = 0
        batch_count = 0
        # per epoch training, do
        for i, data in enumerate(train_loader, 0):
            model.train()
            x, label = data
            if imgFlat:
                x = x.view(-1,imgSize**2)
            x = x.to(C['device'])
            label = label.to(C['device'])
            optimizer.zero_grad()
            if config['noise_std'] > 0:
                generate_variation(model,noise_std=config['noise_std'])
            output = model(x)
            l = lossfunc(output,label)
            l.backward()
            optimizer.step()
            total_loss += l.data.item()
            batch_count += 1
        
        total_loss /= batch_count
        tb.add_scalar('epoch loss',total_loss,epoch+1)
        tb.add_scalar('epoch time',time.time()-start,epoch+1)
        tb.add_scalar('learning rate',current_lr,epoch+1)

        # console output
        if epoch % printPerEpoch == printPerEpoch-1:
            print("epoch %s loss %s [%.4f seconds]"%(epoch+1,total_loss,time.time()-start))

        if C['valPerEp'] is None:
            continue

        # validation
        if epoch % C['valPerEp'] == 0:
            val = evaluate(test_loader,model,noise_std = C['noise_std'],repeat = C['valSample'],imgSize=imgSize,imgFlat=imgFlat,device = C['device'])

            tb.add_scalar('validation/mean accuracy',val['mean_acc'],epoch+1)
            tb.add_scalar('validation/qtl accuracy',val['qtl_acc'],epoch+1)
            tb.add_scalar('validation/mean loss',val['mean_loss'],epoch+1)
            tb.add_scalar('validation/qtl loss',val['qtl_loss'],epoch+1)
            tb.add_scalar('validation/validation time',val['test time'],epoch+1)
            tb.add_histogram('validation/accuracy',val['acc_list'],epoch+1)
            tb.add_histogram('validation/loss',val['loss_list'],epoch+1)

            print("Epoch %s validation [%.4f seconds]"%(epoch+1,val['test time']))
            print("mean acc %.4f mean loss %.4f"%(val['mean_acc'],val['mean_loss']))

    tb.close()