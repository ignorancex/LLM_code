"""This file support evaluates both loss and accuracy over many batches
"""
from src.loss_functions import loss_acc_batch_fn
import torch
import logging
from collections import OrderedDict


def evaluate_model(args, model, tokenizer, data, batch_size):
    model.eval()
    loss = 0.
    accur = 0.
    perpl = 0.
    loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=False)
    logging.info("Evaluating Model")
    count = 0.
    count_tok = 0.
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print("Iteration", i, "of", len(loader))
        results = loss_acc_batch_fn(model, tokenizer, batch, args.task)
        loss += results['nll_sum']
        accur += results['acc_sum']
        count += len(batch['input_ids'])
        count_tok += results["n_tokens"]
    model.train()
    loss = loss / count_tok
    perpl = torch.exp(loss)
    accur = accur / count
    return dict(loss=loss, acc=accur, ppl=perpl)


@torch.no_grad()
def evaluate_models_fn(args, model, tokenizer, train_data, test_data):
    eval_tr = evaluate_model(args, model, tokenizer,
                             train_data, args.batch_size)
    eval_te = evaluate_model(args, model, tokenizer,
                             test_data, args.batch_size)
    ret = OrderedDict()
    if (args.task == "zsre"):
        ret["loss_train"] = eval_tr["loss"]
        ret["loss_test"] = eval_te["loss"]
        ret["acc_train"] = eval_tr["acc"]
        ret["acc_test"] = eval_te["acc"]
    if args.task == "wiki":
        ret["loss_train"] = eval_tr["loss"]
        ret["loss_test"] = eval_te["loss"]
        ret["perpl_train"] = eval_tr["ppl"]
        ret["perpl_test"] = eval_te["ppl"]
    return ret
