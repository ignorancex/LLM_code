import torch
import numpy as np
import torch.nn.functional as F

def energy(logits):
    return torch.logsumexp(logits, dim=1)


def max_logit(logits):
    return torch.amax(logits, dim=1)


def MSP(logits):
    prob = torch.nn.functional.softmax(logits, dim=1)
    return torch.amax(prob, dim=1)


def predict(logits):
    prob = torch.nn.functional.softmax(logits, dim=1)
    _, idx = torch.max(prob, dim=1)
    return idx


def entropy(logits, dim=-1):
    probs = torch.nn.functional.softmax(logits, dim=1)
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent


def max_cosine(eval_features, last_layer, l=1):
    weight = last_layer.weight.cpu()
    result = []

    for i in range(0, eval_features.shape[0], 128):
        with torch.no_grad():
            length = min(128, eval_features.shape[0] - i)
            feat = eval_features[i:(i+length),:]
            cosine = F.cosine_similarity(weight.unsqueeze(0).expand(feat.shape[0], 
                weight.shape[0], weight.shape[1]), feat.unsqueeze(1).expand(feat.shape[0], weight.shape[0], weight.shape[1]), dim=2)

            result += (l*torch.max(cosine, dim=1)[0]).tolist()

    return np.array(result)


def react(training_features, eval_features, percentile, last_layer):
    threshold = np.percentile(
        training_features.cpu().detach().numpy(), percentile)
    eval_features = torch.clip(eval_features, max=threshold).cuda()
    logits = eval_features @ last_layer.weight.T + last_layer.bias
    result = energy(logits).cpu().detach().numpy()
    return result


def ash(x, last_layer, percentile, version):
    assert x.dim() == 2
    x = x.cuda()
    reshaped_x = x.unsqueeze(2).unsqueeze(3)
    if version == 's':
        ash_s(reshaped_x, percentile)
    else:
        ash_b(reshaped_x, percentile)
    logits = x @ last_layer.weight.T + last_layer.bias
    result = energy(logits).cpu().detach().numpy()
    return result

def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x

def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x