# The file Multi_Column.py contains the python based implementation of our work
#"A Genetic Algorithm based Kernel-size Selection Approach for a Multi-column Convolutional Neural Network"

import random
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os.path

# only 3x3, 5x5, 7x7 kernel sizes are used
kernel_choices = [3, 5, 7]
num_epochs = 100
batch_size = 250
learning_rate = 0.001
print_every = 8
best_accuracy = torch.FloatTensor([0])
start_epoch = 0
num_input_channel = 1
num_output_channel = 10
num_classes = 10
resume_weights = "Architecture/checkpoint.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, ],
    #                      std=[0.5, ])
])

num_input_channel = 1


# reshaping the output feature map of the CNN to a shape of 4x4
def Reshape(tensor):
    pad_len = 4 - int(tensor.shape[2])
    paddind1 = nn.ZeroPad2d((1, 0, 1, 0))
    paddind2 = nn.ZeroPad2d(padding=1)

    while pad_len > 0:
        if pad_len == 1:
            tensor = paddind1(tensor)
            pad_len -= 1
        else:
            tensor = paddind2(tensor)
            pad_len -= 2

    return tensor


# Definition class of the network
class Basic_Net(nn.Module):
    def __init__(self):
        super(Basic_Net, self).__init__()

        self.layer11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer21 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=5, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer23 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer31 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=7, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer32 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer33 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # final fc layer

        self.fc1 = nn.Sequential(
            nn.Linear(22784, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(2048, num_output_channel),
            nn.BatchNorm1d(num_output_channel),
            nn.Softmax(dim=1))

    def forward(self, x):
        x1 = self.layer11(x)

        x1 = self.layer12(x1)

        x1 = self.layer13(x1)
        x1 = x1.view(-1, self.num_flat_features(x1))

        # for second column
        x2 = self.layer21(x)

        x2 = self.layer22(x2)

        x2 = self.layer23(x2)
        x2 = x2.view(-1, self.num_flat_features(x2))

        # for third column
        x3 = self.layer31(x)

        x3 = self.layer32(x3)

        x3 = self.layer33(x3)
        x3 = x3.view(-1, self.num_flat_features(x3))

        x = torch.cat((x1, x2), 1)
        x = torch.cat((x, x3), 1)

        # final fc layer

        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


print("Loading the dataset...")
train_set = torchvision.datasets.ImageFolder(root="DataSetName/Train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

val_set = torchvision.datasets.ImageFolder(root="DataSetName/Val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
val_loader2 = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root="DataSetName/Test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
print("Dataset is Loaded")

print("Train Loader: ", len(train_loader))
print("Validation Loader: ", len(val_loader))
print("Test Loader: ", len(test_loader))


# Function for initializing population
def InitPopulation(pop_size, ind_dim):
    pop = []
    popDup = []

    for i in range(0, pop_size):
        pop.append([])
        popDup.append([])

        for j in range(0, ind_dim):
            kernel_value = random.choice(kernel_choices)
            pop[i].append(kernel_value)
            popDup[i].append(kernel_value)

    print("Initial Population: ", pop, end="\n\n")

    return pop


# Function for single point cross-over operation
def SinglePointCrossover(ind1, ind2):
    size = min(len(ind1), len(ind2))

    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


# Function for two point cross-over operation
def TwoPointCrossover(ind1, ind2):
    size = min(len(ind1), len(ind2))

    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)

    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2


# Function for mutation operation performed randomly
def RandomMutation(ind):
    num_index = random.randint(1, len(ind))
    index_list = []

    for i in range(num_index):
        index_list.append(random.randint(0, len(ind)))

    for i in range(len(ind)):
        if i in index_list:
            ind_val = ind[i]
            mutate_val = random.choice(list(set(kernel_choices) - {ind_val}))
            ind[i] = mutate_val

    return ind


# Function for calculating fitness value on the basis of test accuracy
def FitnessValue(model, ind):

    if ind[0] == 3 or 5:
        model.layer11[0].kernel_size = (ind[0], ind[0])
        model.layer11[0].stride = (2, 2)
    else:
        model.layer11[0].kernel_size = (ind[0], ind[0])
        model.layer11[0].stride = (1, 1)

    if ind[1] == 3 or 5:
        model.layer12[0].kernel_size = (ind[1], ind[1])
        model.layer12[0].stride = (2, 2)
    else:
        model.layer12[0].kernel_size = (ind[1], ind[1])
        model.layer12[0].stride = (1, 1)

    if ind[2] == 3 or 5:
        model.layer13[0].kernel_size = (ind[2], ind[2])
        model.layer13[0].stride = (2, 2)
    else:
        model.layer13[0].kernel_size = (ind[2], ind[2])
        model.layer13[0].stride = (1, 1)

    ####################################

    if ind[3] == 3 or 5:
        model.layer21[0].kernel_size = (ind[3], ind[3])
        model.layer21[0].stride = (2, 2)
    else:
        model.layer21[0].kernel_size = (ind[3], ind[3])
        model.layer21[0].stride = (1, 1)

    if ind[4] == 3 or 5:
        model.layer22[0].kernel_size = (ind[4], ind[4])
        model.layer22[0].stride = (2, 2)
    else:
        model.layer22[0].kernel_size = (ind[4], ind[4])
        model.layer22[0].stride = (1, 1)

    if ind[5] == 3 or 5:
        model.layer23[0].kernel_size = (ind[5], ind[5])
        model.layer23[0].stride = (2, 2)
    else:
        model.layer23[0].kernel_size = (ind[5], ind[5])
        model.layer23[0].stride = (1, 1)

    ######################################

    if ind[6] == 3 or 5:
        model.layer31[0].kernel_size = (ind[6], ind[6])
        model.layer31[0].stride = (2, 2)
    else:
        model.layer31[0].kernel_size = (ind[6], ind[6])
        model.layer31[0].stride = (1, 1)

    if ind[7] == 3 or 5:
        model.layer32[0].kernel_size = (ind[7], ind[7])
        model.layer32[0].stride = (2, 2)
    else:
        model.layer32[0].kernel_size = (ind[7], ind[7])
        model.layer32[0].stride = (1, 1)

    if ind[8] == 3 or 5:
        model.layer33[0].kernel_size = (ind[8], ind[8])
        model.layer33[0].stride = (2, 2)
    else:
        model.layer33[0].kernel_size = (ind[8], ind[8])
        model.layer33[0].stride = (1, 1)

    print("Chromosome: ", ind)
    fitness_score = Train_Eval_Epoch(model)
    print("Chromosome Fitness Value: ", fitness_score, end="\n\n")

    return fitness_score


def FindFitnessScore(model, pop):
    print("Calculating fitness value of the chromosomes from the population")
    fitness_list = []
    for i in range(len(pop)):
        fitness_list.append(FitnessValue(model, pop[i]))

    return fitness_list


# Function for parent selection
def GeneSelection(pop, fitness_list):
    fitness_val_list = fitness_list.copy()

    fitness_val_list = sorted(fitness_val_list, reverse=True)

    chrom1_val = fitness_val_list[0]
    chrom2_val = fitness_val_list[1]

    if chrom1_val == chrom2_val:
        index_list = [i for i, n in enumerate(fitness_list) if n == chrom1_val]
        chrom1_index = index_list[0]
        chrom2_index = index_list[1]
    else:
        chrom1_index = fitness_list.index(chrom1_val)
        chrom2_index = fitness_list.index(chrom2_val)

    chromosome1 = pop[chrom1_index].copy()
    chromosome2 = pop[chrom2_index].copy()

    print("Selected chromosomes for genetic process: first parent-> {}, second parent-> {}".format(chromosome1, chromosome2))

    return chromosome1, chromosome2


# Function for performing genetic process stepwise
def GeneticProcess(model, pop, fitness_list):
    ind1, ind2 = GeneSelection(pop, fitness_list)
    ind1, ind2 = SinglePointCrossover(ind1, ind2)
    ind1 = RandomMutation(ind1)
    ind2 = RandomMutation(ind2)

    fitness_ind1 = FitnessValue(model, ind1)
    fitness_ind2 = FitnessValue(model, ind2)

    if fitness_ind1 >= fitness_ind2:
        max_fit_val = fitness_ind1
        max_fit_ind = ind1
    else:
        max_fit_val = fitness_ind2
        max_fit_ind = ind2

    return max_fit_val, max_fit_ind


# Function for updating better fit individuals to the initial population
def UpdatePopulation(model, pop, fitness_list):
    fitness_value_list = fitness_list.copy()

    fitness_value_list = sorted(fitness_value_list)
    minfit_chrom = fitness_value_list[0]

    replace_val, replace_chrom = GeneticProcess(model, pop, fitness_list)

    minfit_index = fitness_list.index(minfit_chrom)

    pop[minfit_index] = replace_chrom
    fitness_list[minfit_index] = replace_val

    print("Updated Population: ", pop)

    return pop, fitness_list


# Function for training the neural network
def Train(model, optimizer, trainLoader, loss_fun):
    average_time = 0
    total = 0
    correct = 0
    model.train()
    for i, (images, labels) in enumerate(trainLoader):
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        average_time += batch_time

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# Function for finding accuracy on validation-set or test-set
def Eval(model, testLoader):
    model.eval()

    total = 0
    correct = 0
    for i, (data, labels) in enumerate(testLoader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


# Saving the network parameters (weights and biases)
def save_checkpoint(state, is_best, filename="Architecture/checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)


# Function for performing training and evaluation operation
def Train_Eval_Epoch(model):
    global learning_rate, best_accuracy, start_epoch, epoch
    for epoch in range(num_epochs):

        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        if learning_rate > 0.00001:
            learning_rate = learning_rate * 0.05

        Train(model, optimizer, train_loader, criterion)
        val_acc = Eval(model, val_loader)

        acc = torch.FloatTensor([val_acc])

        is_best = bool(acc.numpy() > best_accuracy.numpy())

        best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best)

    if os.path.isfile(resume_weights):
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])

    test_acc = Eval(model, test_loader)
    return test_acc


def main():
    global count
    count = 0

    global maxGen
    maxGen = 20

    global popSize
    popSize = 10

    global indDim
    indDim = 9

    model = Basic_Net()
    if cuda:
        model.cuda()

    pop = InitPopulation(popSize, indDim)
    fitness_list = FindFitnessScore(model, pop)

    for gen in range(maxGen):
        print("Generation No: [%d/%d]" % (gen+1, maxGen))

        model = Basic_Net()
        if cuda:
            model.cuda()

            pop, fitness_list = UpdatePopulation(model, pop, fitness_list)

    maxFitVal = max(fitness_list)
    maxIndex = fitness_list.index(maxFitVal)

    maxFitInd = pop[maxIndex]

    print("Optimal kernel combination: {}, Accuracy: {}".format(maxFitInd, maxFitVal))


if __name__ == "__main__":
    main()
