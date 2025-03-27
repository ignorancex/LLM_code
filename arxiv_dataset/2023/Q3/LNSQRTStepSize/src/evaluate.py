"""
Evaluating the model on the test set.
"""

import torch
import tqdm

def evaluate(args,test_loader, net, criterion, device):  
    """
    Args:
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.
        device: denoting using CPU or GPU.

    Outputs:
        Average loss and accuracy achieved by the model in the test set.
    """    
    net.eval()

    accurate = 0
    loss = 0.0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
        #for data in test_loader:
            images, labels = data
            if args.dataset == 'Flower102':
                labels = torch.sub(labels, 1)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            
            #testnan=torch.isnan(outputs)
            #for x in testnan:
                #for y in x:
                    #if y:
                        #raise ValueError("output is nan")
                    
            loss += criterion(outputs, labels) * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            accurate += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 1.0 * accurate / total
        loss = loss.item() / total
        if(total==0):
            print("eroorrrrrr")

    return (loss, accuracy)
