import torch
from torch.autograd import Variable


# function is heavily taken from original ODIN code
def odin(inputs, model, epsilon, temperature):
    model.zero_grad()
    inputs.requires_grad = True
    logits = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    maxIndexTemp = torch.argmax(logits).item()
    labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
    logits = logits / temperature
    loss = criterion(logits, labels)
    loss.backward()
    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0).cuda()
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
    gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
    # Adding small perturbations to images
    tempInputs = inputs.data - epsilon * gradient
    outputs = model(Variable(tempInputs))
    outputs = outputs / temperature
    return torch.amax(outputs)


def odin_batched(inputs, model, epsilons, temperature):
    model.zero_grad()
    inputs.requires_grad = True
    logits = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    maxIndexTemp = torch.argmax(logits).item()
    labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
    logits = logits / temperature
    loss = criterion(logits, labels)
    loss.backward()
    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0).cuda()
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
    gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
    # Adding small perturbations to images
    tempInputs = []
    for e in epsilons:
        tempInputs.append(inputs.data - e * gradient)
    tempInputs = torch.cat(tempInputs)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temperature
    return torch.amax(outputs, dim=1)
