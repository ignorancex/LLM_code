import torch
import torch.nn as nn
import sys
#sys.path.append('auto-attack')
#from autoattack import AutoAttack
#sys.path.append('robustness')
#from robustness import model_utils

from data import make_gtsrb_testloader

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

### clean accuracy

def test():
    #classifier = torch.load("checkpoint/net.pth")
    classifier = torch.load("checkpoint/pretrained.pth")

    testloader = make_gtsrb_testloader()

    classifier.eval()
    correct = 0
    total = 0
    for inputs, labels in testloader:
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs, dim=1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)

    print("gtsrb test: %.1f%% (%d / %d)" % (correct / total * 100, 
        correct, total))

    return correct

#### standard PGD
#
#class _FakeDS(object):
#    mean = torch.tensor([0])
#    std = torch.tensor([1])
#
#def make_pgd_adversary(net):
#    m, _ = model_utils.make_and_restore_model(arch=net, dataset=_FakeDS)
#    return m
#
#def test_pgd():
#    #classifier = torch.load("checkpoint/net.pth")
#    classifier = torch.load("checkpoint/pretrained.pth")
#    adversary = make_pgd_adversary(classifier)
#    if use_cuda:
#        classifier = classifier.to(device)
#        adversary = adversary.to(device)
#
#    testloader = make_gtsrb_testloader()
#
#    classifier.eval()
#    adversary.eval()
#    correct = 0
#    total = 0
#    for inputs, labels in testloader:
#        if use_cuda:
#            inputs = inputs.to(device)
#            labels = labels.to(device)
#        outputs, final_inp = adversary(inputs, make_adv=True, target=labels,
#            constraint="inf", step_size=1/255, eps=8/255, 
#            iterations=20, random_start=True, random_restarts=10, 
#            use_best=True)
#        _, predicted = torch.max(outputs, dim=1)
#        correct += predicted.eq(labels).sum().cpu().item()
#        total += inputs.size(0)
#
#    print("gtsrb pgd: %.1f%% (%d / %d)" % (correct / total * 100, 
#        correct, total))
#
#    return correct
#
#### auto attack 
#
#def test_auto():
#    #classifier = torch.load("checkpoint/net.pth")
#    classifier = torch.load("checkpoint/pretrained.pth")
#    if use_cuda:
#        classifier = classifier.to(device)
#    adversary = AutoAttack(classifier, norm='Linf', eps=8/255, verbose=False)
#
#    testloader = make_gtsrb_testloader()
#
#    classifier.eval()
#    correct = 0
#    total = 0
#    for inputs, labels in testloader:
#        if use_cuda:
#            inputs = inputs.to(device)
#            labels = labels.to(device)
#        adv = adversary.run_standard_evaluation(inputs, labels, bs=500)
#        outputs = classifier(adv)
#        _, predicted = torch.max(outputs, dim=1)
#        correct += predicted.eq(labels).sum().cpu().item()
#        total += inputs.size(0)
#
#    print("gtsrb test: %.1f%% (%d / %d)" % (correct / total * 100, 
#        correct, total))
#
#    return correct

if __name__ == "__main__":
    test()
    #test_pgd()
    #test_auto()

