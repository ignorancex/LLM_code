import copy
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
from transformers import BartModel, BartConfig, BartForConditionalGeneration
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            weight_decay=0,
            adam_eps=1e-9,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))
    return optim


class FineTuneModel(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(FineTuneModel, self).__init__()
        self.args = args
        self.finetune = args.finetune_bart

        temp_dir = args.temp_dir
        config = BartConfig(max_position_embeddings=args.max_pos,
                                    dropout=args.dropout,
                                    attention_dropout=args.attention_dropout)
        if(args.large):
            self.model = BartForConditionalGeneration(config).from_pretrained('facebook/bart-large', cache_dir=temp_dir)
        else:
            self.model = BartForConditionalGeneration(config).from_pretrained('facebook/bart-base', cache_dir=temp_dir)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        # initialize parameters

        self.to(device)


    def forward(self, src, tgt, mask_src, mask_tgt, labels):
        if self.finetune:
            res = self.model(input_ids=src,
                    attention_mask=mask_src,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=mask_tgt,
                    labels=labels)
            return res.loss, res.logits
        else:
            self.eval()
            with torch.no_grad():
                res = self.model(input_ids=src,
                    attention_mask=mask_src,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=mask_tgt)
            return res.logits

    def generate(self, src, beam_size, max_length, min_length, early_stopping=True):
        hypos = self.model.generate(src,
                    num_beams=beam_size,
                    max_length=max_length,
                    min_length=min_length,
                    early_stopping=early_stopping)
        return hypos
