import torch
import torch.nn as nn
import torchvision.utils as utils
import torch.optim as optim
from torchvision import transforms
from data import normalize, make_picto_trainloader
from model import Classifier

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu') 
host = torch.device('cpu')

contexts = None
def create_inputs(model, inputs, masks, labels, criterion, iters=8,
        beps=16, feps=4):
    global contexts

    chans = inputs.shape[1]
    masks = torch.stack([masks] * chans)
    dims = list(range(len(inputs.shape)))
    dims[0] = 1
    dims[1] = 0
    masks = masks.permute(dims)

    # reset contexts
    if contexts is not None:
        reset = torch.randn(contexts.shape[0])
        contexts[reset > 0.95] = inputs[reset > 0.95]

    # observe inputs
    refined = inputs.clone()
    if contexts is not None:
        refined[masks < 1] = contexts[masks < 1]

    # perform local refinement
    contexts = refined.clone().detach_()
    training = model.training
    model.train(False)
    beps, feps = beps / 255, feps / 255
    for it in range(iters):
        refined.requires_grad = True
        outputs = model(refined)

        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        grad = refined.grad
        with torch.no_grad():
            sign = grad.sign()
            norm = grad / torch.max(torch.abs(grad))
            step = ((iters - it) * sign + it * norm) / iters
            step = beps / iters * step

            contexts = contexts + step
            contexts = torch.clamp(contexts, min=0, max=1).detach_()

            refined = refined + step 
            diff = torch.clamp(refined - inputs, min=-feps, max=feps)
            refined[masks >= 1] = inputs[masks >= 1] + diff[masks >= 1]
            refined = torch.clamp(refined, min=0, max=1).detach_()
    refined = refined.to(host)
    for i in range(len(refined)):
        refined[i] = normalize(refined[i])
    refined = refined.to(device)
    model.train(training)
    return refined

def train(epochs=300, super_batch=5, num_batches=20):
    net = Classifier()
    net.train()

    optimizer = optim.Adam(net.parameters(),
            lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net.to(device)

    trainloader = make_picto_trainloader()
    real_correct = 0
    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))

        factor = min(1, max(0, (epoch - 2) / 160))
        trainloader.dataset.adjust_alpha(factor)

        print("factor: %.3f" % factor)

        correct = 0
        total = 0
        total_loss = 0
        for nb in range(num_batches):
            loss = 0
            for sb in range(super_batch):
                for inputs, masks, labels in trainloader:
                    if use_cuda:
                        inputs = inputs.to(device)
                        masks = masks.to(device)
                        labels = labels.to(device)

                    inputs = create_inputs(net, inputs, masks, labels, 
                            criterion)
                    outputs = net(inputs) 
                    batch_loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs, dim=1)
                    correct += predicted.eq(labels).sum().item()
                    total += inputs.size(0)

                    total_loss += batch_loss
                    loss += batch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("train: %.1f%% (%d / %d) | loss: %.3f" % (correct / total * 100, 
            correct, total, total_loss))

    net.to(host)
    torch.save(net, "checkpoint/net.pth")

if __name__ == "__main__":
    train()

