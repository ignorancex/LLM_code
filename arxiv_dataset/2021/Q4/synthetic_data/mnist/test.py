import torch
from data import make_mnist_testloader

def test():
    #net = torch.load("checkpoint/net.pth")
    net = torch.load("checkpoint/pretrained.pth")

    mnist_testloader = make_mnist_testloader()

    net.eval()
    correct = 0
    total = 0
    for inputs, labels in mnist_testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, dim=1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)

    print("test mnist: %.1f%% (%d / %d)" % (correct / total * 100, 
        correct, total))

    return correct, total

if __name__ == "__main__":
    test()


