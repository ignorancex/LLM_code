import torch
import torch.nn as nn
import torch.optim as optim
from data import make_font_trainloader
from model import Net 

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu') 
host = torch.device('cpu')

contexts = None
def create_inputs(model, objects, labels, criterion, iters=8, eps=32):
    global contexts

    # reset contexts
    noise = torch.rand(objects.shape) / 2
    reset = torch.randn(objects.shape[0])
    if use_cuda:
        noise = noise.to(device)
        reset = reset.to(device)

    if contexts is None:
        contexts = noise
    else:
        contexts[reset > 0.8] = noise[reset > 0.8]

    # observe inputs
    inputs = (2 * objects + contexts) / 3

    # perform local refinement
    eps = eps / 255
    inputs.detach_()

    training = model.training
    model.train(False)

    for it in range(iters):
        inputs.requires_grad = True
        outputs = model(inputs)

        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        grad = inputs.grad
        with torch.no_grad():
            sign = grad.sign()
            norm = grad / torch.max(torch.abs(grad))
            step = ((iters - it) * sign + it * norm) / iters
            step = eps / iters * step / 2.5

            inputs = inputs + step
            inputs = torch.clamp(inputs, min=0, max=1).detach_()

    contexts = inputs.clone().detach_()
    return inputs


def train(epochs=300, super_batch=5, num_batches=20):
    net = Net()
    net.train()

    optimizer = optim.Adam(net.parameters(),
            lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net.to(device)

    real_correct = 0

    trainloader = make_font_trainloader()
    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))

        factor = min(1, max(0, (epoch - 2) / 50))
        trainloader.dataset.adjust_alpha(factor)

        print("factor: %.3f" % factor)

        correct = 0
        total = 0
        total_loss = 0
        for nb in range(num_batches):
            loss = 0
            for sb in range(super_batch):
                for objects, labels in trainloader:
                    if use_cuda:
                        objects = objects.to(device)
                        labels = labels.to(device)

                    inputs = create_inputs(net, objects, labels, criterion)
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

