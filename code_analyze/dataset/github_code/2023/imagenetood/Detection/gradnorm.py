import torch


# function based on code from original gradnorm paper
def gradnorm(inputs, model, temperature, num_classes, last_layer):
    model.zero_grad()
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(inputs)
    targets = torch.ones((inputs.shape[0], num_classes)).cuda()
    logits = logits / temperature
    loss = torch.mean(torch.sum(-targets * logsoftmax(logits), dim=-1))
    loss.backward()
    layer_grad = last_layer.weight.grad.data
    return torch.sum(torch.abs(layer_grad)).item()
