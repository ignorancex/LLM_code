import torch
from torch import nn
import logging
# adamw?
def getOpt(params,config):
    dic={}
    dic['params']=params
    dic['initial_lr']=config.trainParam.learnRate
    if config.trainParam.learnType == "Adam":
        return torch.optim.AdamW([dic], lr = config.trainParam.learnRate, weight_decay = config.trainParam.adam_weight_decay)
    elif config.trainParam.learnType == "SGD":
        return torch.optim.SGD([dic], lr=config.trainParam.learnRate, momentum=config.trainParam.sgd_momentum,nesterov=True)
    else:
        logging.error("Cannot resolve optimizer type %s"%config.trainParam.learnType)

def getSch(optimizer,config):
    lambda1Str = "lambda epoch: "+config.trainParam.learnRateMulti
    lambda1 = eval(lambda1Str)
    fromEpoch=-1
    if config.continueTrain.enableContinue > 0:
        fromEpoch = config.continueTrain.fromEpoch
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1, last_epoch=fromEpoch)
