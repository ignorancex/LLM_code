import json
import os
import time
import torch
import torch.nn as nn

from torch.autograd import Variable

from vqa_utils import VqaUtils, PerTypeMetric
from metrics import Metrics, accumulate_metrics
import numpy as np
from tqdm import tqdm

def compute_score_with_logits(preds, labels, logits_key='logits'):
    """
    Computes softscores
    :param logits:
    :param labels:
    :return:
    """
    logits = preds[logits_key]
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        one_hots = one_hots.cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def save_metrics_n_model(metrics, model, optimizer, args, is_best):
    """
    Saves all the metrics, parameters of models and parameters of optimizer.
    If current score is the highest ever, it also saves this model as the best model
    """
    metrics_n_model = metrics.copy()
    metrics_n_model["model_state_dict"] = model.state_dict()
    metrics_n_model["optimizer_state_dict"] = optimizer.state_dict()
    metrics_n_model["args"] = args

    with open(os.path.join(args.expt_save_dir, 'latest-model.pth'), 'wb') as lmf:
        torch.save(metrics_n_model, lmf)

    if is_best:
        with open(os.path.join(args.expt_save_dir, 'best-model.pth'), 'wb') as bmf:
            torch.save(metrics_n_model, bmf)

    return metrics_n_model


def train(model, train_loader, val_loader, num_epochs, optimizer, criterion, args, start_epoch=0, best_val_score=0,
          best_val_epoch=0):
    """
    This is the main training loop. It trains the model, evaluates the model and saves the metrics and predictions.
    """
    metrics_stats_list = []
    val_per_type_metric_list = []

    if args.apply_rubi:
        val_per_type_metric_list_rubi, val_per_type_metric_list_q = [], []

    lr_decay_step = 2
    lr_decay_rate = .25
    if optimizer is None:
        
        # lr_decay_epochs = range(10, 25, lr_decay_step)
        # gradual_warmup_steps = [0.5 * args.lr, 1.0 * args.lr, 1.5 * args.lr, 2.0 * args.lr]
        # if args.apply_rubi:
        lr_decay_epochs = range(14, 100, lr_decay_step)
        gradual_warmup_steps = [i * args.lr for i in torch.linspace(0.5, 2.0, 7)]
        print(gradual_warmup_steps)
        # else:
        #     lr_decay_epochs = range(10, 25, lr_decay_step)
        #     gradual_warmup_steps = [0.5 * args.lr, 1.0 * args.lr, 1.5 * args.lr, 2.0 * args.lr]
        optimizer = getattr(torch.optim, args.optimizer)(filter(lambda p: p.requires_grad, model.parameters()),
                                                         lr=args.lr)
    else:
        gradual_warmup_steps = []
        lr_decay_epochs = range(14, 100, lr_decay_step)



    iter_num = 0
    if args.test and start_epoch == num_epochs:
        start_epoch = num_epochs - 1
    for epoch in range(start_epoch, num_epochs):
        if epoch < len(gradual_warmup_steps):
            optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
        elif epoch in lr_decay_epochs:
            optimizer.param_groups[0]['lr'] *= lr_decay_rate
        else:
            optimizer.param_groups[0]['lr'] = args.lr
        print("lr {}".format(optimizer.param_groups[0]['lr']))

        is_best = False
        train_metrics, val_metrics = Metrics(), Metrics()

        if args.apply_rubi:
            train_metrics_rubi, val_metrics_rubi = Metrics(), Metrics()
            train_metrics_q, val_metrics_q = Metrics(), Metrics()
        else:
            val_metrics_rubi, val_metrics_q = None, None

        if not args.test:
            tqdm_train_loader = tqdm(train_loader, position=0, leave=True)
            for i, (visual_features, boxes, question_features, answers, question_types, question_ids,
                    question_lengths) in enumerate(tqdm_train_loader):

                tqdm_train_loader.set_description(f'Loss : {train_metrics.get_loss()} | Score {train_metrics.get_score()}')

                visual_features = Variable(visual_features.float())
                boxes = Variable(boxes.float())
                question_features = Variable(question_features)
                answers = Variable(answers)

                if torch.cuda.is_available():
                    visual_features = visual_features.cuda()
                    boxes = boxes.cuda()
                    question_features = question_features.cuda()
                    answers = answers.cuda()



                pred = model(visual_features, boxes, question_features, answers, question_lengths)
                loss = criterion(pred, answers)['loss']
                loss.backward()
                train_metrics.update_per_batch(model, answers, loss, pred, visual_features.shape[0])
                if args.apply_rubi:
                    train_metrics_rubi.update_per_batch(model, answers, loss, pred, visual_features.shape[0],
                                                        logits_key='logits_rubi')
                    train_metrics_q.update_per_batch(model, answers, loss, pred, visual_features.shape[0],
                                                     logits_key='logits_q')
                nn.utils.clip_grad_norm_(model.parameters(), 50)
                optimizer.step()
                optimizer.zero_grad()
                iter_num += 1
                #if i % 10 == 0:
                    #train_metrics.print(epoch)
                    # if args.apply_rubi:
                    #     print("\n\n### logits_rubi ###")
                    #     train_metrics_rubi.print(epoch)
                    #     print("\n\n### logits_q ###")
                    #     train_metrics_q.print(epoch)
            train_metrics.update_per_epoch()
            if args.apply_rubi:
                train_metrics_rubi.update_per_epoch()
                train_metrics_q.update_per_epoch()

        if None != val_loader:  # TODO: "val_loader is not None' was not working for some reason
            print("Starting the test ... ")

            model.eval()
            with torch.no_grad():
                val_results = evaluate_by_logits_key(model, val_loader, epoch, criterion, args, val_metrics,
                                                     logits_key='logits')
                if args.apply_rubi:
                    val_results_rubi = evaluate_by_logits_key(model, val_loader, epoch, criterion, args,
                                                              val_metrics_rubi,
                                                              logits_key='logits_rubi')
                    val_results_q = evaluate_by_logits_key(model, val_loader, epoch, criterion, args, val_metrics_q,
                                                           logits_key='logits_q')
                # eval_results = evaluate(model, val_loader, epoch, criterion, args, val_metrics, val_metrics_rubi,
                #                         val_metrics_q) # TODO: FIX, use a loop to do this

            model.train()
            if val_metrics.score > best_val_score:
                best_val_score = val_metrics.score
                best_val_epoch = epoch
                is_best = True

            save_val_metrics = not args.test or not args.test_does_not_have_answers
            if save_val_metrics:
                print("Best val score {} at epoch {}".format(best_val_score, best_val_epoch))
                print(f"### Val from Logits {val_metrics.score}")
                if args.apply_rubi:
                    print(f"### Val from Logits_rubi {val_metrics_rubi.score}")
                    print(f"### Val from Logits_q {val_metrics_q.score}")
                    # print(
                    #     f"##### by logits key {val_metrics_by_logits_key.score} "
                    #     f"val_metrics_by_logits_key_rubi {val_metrics_by_logits_key_rubi.score} "
                    #     f"Logits score: {val_metrics.score} "
                    #     f"Logits_rubi score: {val_metrics_rubi.score} "
                    #     f"Logits_q score: {val_metrics_q.score} ####")

                val_per_type_metric_list.append(val_results['per_type_metric'].get_json())
                if args.apply_rubi:
                    val_per_type_metric_list_rubi.append(val_results_rubi['per_type_metric'].get_json())
                    val_per_type_metric_list_q.append(val_results_q['per_type_metric'].get_json())

            metrics = accumulate_metrics(epoch, train_metrics, val_metrics, val_results['per_type_metric'],
                                         best_val_score, best_val_epoch,
                                         save_val_metrics)

            metrics_stats_list.append(metrics)

            # Add metrics + parameters of the model and optimizer
            metrics_n_model = save_metrics_n_model(metrics, model, optimizer, args, is_best)
            VqaUtils.save_stats(metrics_stats_list, val_per_type_metric_list, val_results['all_preds'],
                                args.expt_save_dir,
                                split=args.test_split, epoch=epoch)
            # if args.apply_rubi:
            #     VqaUtils.save_stats(metrics_stats_list, val_per_type_metric_list_rubi, val_results_rubi['all_preds'],
            #                         args.expt_save_dir,
            #                         split=args.test_split, epoch=epoch, suffix='rubi')
            #     VqaUtils.save_stats(metrics_stats_list, val_per_type_metric_list_q, val_results_q['all_preds'],
            #                         args.expt_save_dir,
            #                         split=args.test_split, epoch=epoch, suffix='q')

        if args.test:
            VqaUtils.save_preds(val_results['all_preds'], args.expt_save_dir, args.test_split, epoch)
            print("Test completed!")
            break


#
#
def evaluate_by_logits_key(model, dataloader, epoch, criterion, args, val_metrics, logits_key='logits'):
    per_type_metric = PerTypeMetric(epoch=epoch)
    with open(os.path.join(args.data_root, args.feature_subdir, 'answer_ix_map.json')) as f:
        answer_ix_map = json.load(f)

    all_preds = []

    for visual_features, boxes, question_features, answers, question_types, question_ids, question_lengths in iter(
            dataloader):
        visual_features = Variable(visual_features.float())
        boxes = Variable(boxes.float())
        question_features = Variable(question_features)

        if torch.cuda.is_available():
            visual_features = visual_features.cuda()
            boxes = boxes.cuda()
            question_features = question_features.cuda()

        if not args.test or not args.test_does_not_have_answers:
            if torch.cuda.is_available():
                answers = answers.cuda()

        pred = model(visual_features, boxes, question_features, None, question_lengths)

        if not args.test or not args.test_does_not_have_answers:
            loss = criterion(pred, answers)['loss']
            val_metrics.update_per_batch(model, answers, loss, pred, visual_features.shape[0], logits_key=logits_key)

        pred_ans_ixs = pred[logits_key].max(1)[1]

        # Create predictions file
        for curr_ix, pred_ans_ix in enumerate(pred_ans_ixs):
            pred_ans = answer_ix_map['ix_to_answer'][str(int(pred_ans_ix))]
            all_preds.append({
                'question_id': int(question_ids[curr_ix].data),
                'answer': str(pred_ans)
            })
            if not args.test or not args.test_does_not_have_answers:
                per_type_metric.update_for_question_type(question_types[curr_ix],
                                                         answers[curr_ix].cpu().data.numpy(),
                                                         pred[logits_key][curr_ix].cpu().data.numpy())
    val_metrics.update_per_epoch()
    return {
        'all_preds': all_preds,
        'per_type_metric': per_type_metric
    }
    # return all_preds, per_type_metric


def _internal_evaluation(args,
                         criterion,
                         pred,
                         answers,
                         model,
                         visual_features,
                         answer_ix_map,
                         question_ids,
                         question_types,
                         _val_metrics,
                         _per_type_metric,
                         _all_preds,
                         suffix=''):
    if not args.test or not args.test_does_not_have_answers:
        loss = criterion(pred, answers)['loss' + suffix]
        _val_metrics.update_per_batch(model, answers, loss, pred, visual_features.shape[0],
                                      logits_key='logits' + suffix)

    pred_ans_ixs = pred['logits' + suffix].max(1)[1]

    # Create predictions file
    for curr_ix, pred_ans_ix in enumerate(pred_ans_ixs):
        pred_ans = answer_ix_map['ix_to_answer'][str(int(pred_ans_ix))]
        _all_preds.append({
            'question_id': int(question_ids[curr_ix].data),
            'answer': str(pred_ans)
        })
        if not args.test or not args.test_does_not_have_answers:
            _per_type_metric.update_for_question_type(question_types[curr_ix],
                                                      answers[curr_ix].cpu().data.numpy(),
                                                      pred['logits' + suffix][curr_ix].cpu().data.numpy())

    _val_metrics.update_per_epoch()


def evaluate(model, dataloader, epoch, criterion, args, val_metrics, val_metrics_rubi=None, val_metrics_q=None):
    with open(os.path.join(args.data_root, args.feature_subdir, 'answer_ix_map.json')) as f:
        answer_ix_map = json.load(f)

    per_type_metric = PerTypeMetric(epoch=epoch)
    all_preds = []
    if val_metrics_rubi is not None:
        per_type_metric_rubi, per_type_metric_q = PerTypeMetric(epoch=epoch), PerTypeMetric(epoch=epoch)
        all_preds_rubi, all_preds_q = [], []

    for visual_features, boxes, question_features, answers, question_types, question_ids, question_lengths in iter(
            dataloader):
        visual_features = Variable(visual_features.float())
        boxes = Variable(boxes.float())
        question_features = Variable(question_features)

        if torch.cuda.is_available():
            visual_features = visual_features.cuda()
            boxes = boxes.cuda()
            question_features = question_features.cuda()

        if not args.test or not args.test_does_not_have_answers:
            if torch.cuda.is_available():
                answers = answers.cuda()

        pred = model(visual_features, boxes, question_features, None, question_lengths)
        _internal_evaluation(args,
                             criterion,
                             pred,
                             answers,
                             model,
                             visual_features,
                             answer_ix_map,
                             question_ids,
                             question_types, val_metrics, per_type_metric, all_preds, suffix='')
        if val_metrics_rubi is not None:
            _internal_evaluation(args,
                                 criterion,
                                 pred,
                                 answers,
                                 model,
                                 visual_features,
                                 answer_ix_map,
                                 question_ids,
                                 question_types, val_metrics_rubi, per_type_metric_rubi, all_preds_rubi, suffix='_rubi')
            _internal_evaluation(args,
                                 criterion,
                                 pred,
                                 answers,
                                 model,
                                 visual_features,
                                 answer_ix_map,
                                 question_ids,
                                 question_types, val_metrics_q, per_type_metric_q, all_preds_q, suffix='_q')

    if val_metrics_rubi is None:
        return {
            'all_preds': all_preds,
            'per_type_metric': per_type_metric
        }
    else:
        return {
            'all_preds': all_preds,
            'per_type_metric': per_type_metric,
            'all_preds_rubi': all_preds_rubi,
            'per_type_metric_rubi': per_type_metric_rubi,
            'all_preds_q': all_preds_q,
            'per_type_metric_q': per_type_metric_q
        }
