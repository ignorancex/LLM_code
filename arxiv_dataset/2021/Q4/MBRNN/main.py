import logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)

import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import model as Network
import PS1
import utils
from anchor_loss import AnchorLoss
from checkpoint import Checkpoint
from optim import Optimizer


def get_dataset_config(batch_size):
    dparams = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2}
    evdparams = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 2}

    return dparams, evdparams


def prepare_db(opt):
    db = {}
    dparams, evdparams = get_dataset_config(opt.batch_size)

    if opt.train:
        train_set = PS1.Dataset('train', opt)
        val_set = PS1.Dataset('val', opt)

        train_loader = torch.utils.data.DataLoader(train_set, **dparams)
        val_loader = torch.utils.data.DataLoader(val_set, **evdparams)

        db['train'] = train_loader
        db['val'] = val_loader
    elif opt.infer:
        infer_set = PS1.Dataset('infer', opt)
        infer_loader = torch.utils.data.DataLoader(infer_set, **evdparams)
        db['infer'] = infer_loader
    else:
        test_set = PS1.Dataset('test', opt)

        test_loader = torch.utils.data.DataLoader(test_set, **evdparams)

        db['test'] = test_loader

    return db


def prepare_loss(opt):
    opt.criterion = AnchorLoss(gamma=opt.gamma)

    return opt


def prepare_model(opt):
    opt.ninp = 17
    model = Network.MBRNN(opt)
    if not opt.train:
        model.eval()

    return model


def prepare_optim(model):
    # setting optimizer
    optimizer = Optimizer(
        torch.optim.Adam(
            model.parameters(),
            lr=0.0008,
            betas=(0.5, 0.999),
            weight_decay=5e-5),
        max_grad_norm=5)
    # setting scheduler of optimizer for learning rate decay.
    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,
        patience=5,
        factor=0.5,
        min_lr=0.000001)
    optimizer.set_scheduler(scheduler)

    return optimizer


def loss_dictionaries(opt):
    ldic = {}
    # *_loss_every: [avg_*_loss2]
    ldic['tloss_pevery'] = []
    ldic['vloss_pevery'] = []
    # tot_*_group_loss: tot_*_print_loss
    ldic['tgloss'] = np.array([0.])
    ldic['vgloss'] = np.array([0.])
    # tot_*_epoch_loss: [tot_*_epoch_loss, count_for_average]
    ldic['teloss'] = [0., 0.]
    ldic['veloss'] = [0., 0.]

    return ldic


def train(db, model, optim, opt):
    logging.info("Train started")

    train_generator, val_generator = db['train'], db['val']
    train_set = train_generator.dataset

    model.train(True)
    model.to(opt.device)

    step = 0  # The number of backpropagation
    stime = time.time()  # start time
    val_min_tloss = np.inf  # minimum validation loss placeholder
    ldic = loss_dictionaries(opt)
    max_training_epoch = opt.max_training_epoch
    batch_max_epoch = int(len(train_set)/opt.batch_size) + 1
    for epoch in range(max_training_epoch):
        for Be, (local_batch, local_zbin) in enumerate(train_generator):
            local_batch = local_batch.to(opt.device)
            local_zbin = local_zbin.to(opt.device)

            optim.zero_grad()
            out_probs = model(local_batch)

            loss = opt.criterion(out_probs, local_zbin)

            # update loss and its denominator
            ldic['teloss'][0] += loss
            ldic['teloss'][1] += 1
            ldic['tgloss'][0] += loss.item()

            loss.backward()
            optim.step()

            if step != 0 and step % opt.pevery == 0:
                mean_tgloss = ldic['tgloss'][0] / opt.pevery
                ldic['tloss_pevery'].append(mean_tgloss)
                ldic['tgloss'] = np.array([0.])

                # for log messages
                prog = float(epoch)/(max_training_epoch)*100
                for param in optim.param_groups():
                    lr = param['lr']  # learning rate.

                # log messages
                log_msg = "Step: %d/%d, " % (Be, batch_max_epoch)
                log_msg += "Progress %d%%, " % prog
                log_msg += "cls loss: %.5f, " % mean_tgloss
                log_msg += "learning rate: %.6f" % lr
                logging.info(log_msg)

            if step != 0 and step % opt.vevery == 0:
                model.eval()
                with torch.set_grad_enabled(False):
                    val_step, val_correct = 0, 0
                    for local_batch, local_zbin in val_generator:
                        local_batch = local_batch.to(opt.device)
                        local_zbin = local_zbin.to(opt.device)

                        out_probs = model(local_batch)

                        val_loss = opt.criterion(out_probs, local_zbin)

                        # get the index of the maximum probability
                        val_pred = out_probs.data.max(1, keepdim=True)[1]
                        val_correct += val_pred.eq(
                            local_zbin.data.view_as(val_pred)).cpu().sum()

                        # update total validation loss
                        # and its denominator
                        ldic['veloss'][0] += val_loss
                        ldic['veloss'][1] += 1
                        ldic['vgloss'][0] += val_loss.item()

                        val_step += 1

                    avg_vgloss = ldic['vgloss'][0]/val_step
                    ldic['vloss_pevery'].append(avg_vgloss)
                    ldic['vgloss'] = np.array([0.])

                    if avg_vgloss < val_min_tloss:
                        val_min_tloss = avg_vgloss
                        checkpoint = Checkpoint(
                            step, epoch, model, optim, opt=opt)
                        checkpoint.save()

                    # for validation log message
                    vdat_len = len(val_generator.dataset)
                    val_acc = float(val_correct)/vdat_len

                    log_msg = "Validation set accuracy: %i/%i (%.6f)\n" % \
                        (val_correct, vdat_len, val_acc)
                    log_msg += "current validation cls loss: %.5f, " % \
                        avg_vgloss
                    log_msg += "current minimum loss: %.5f\n" % val_min_tloss
                    logging.info(log_msg)
                model.train(True)
            step += 1

        # averaged epoch losses for training and validation
        avg_teloss = ldic['teloss'][0]/ldic['teloss'][1]
        avg_veloss = ldic['veloss'][0]/ldic['veloss'][1]
        # initialize total epoch loss dictionries
        ldic['teloss'] = [0., 0.]
        ldic['veloss'] = [0., 0.]

        if epoch >= opt.lr_decay_epoch:
            optim.update(avg_vgloss, epoch)

        log_msg = "Finished epoch %d, " % epoch
        log_msg += "train loss: %.5f, " % avg_teloss
        log_msg += "validation loss: %.5f, " % avg_veloss
        logging.info(log_msg)

    etime = time.time()  # end time
    dur = etime - stime  # training time
    logging.info("Training is done. Took %.3fh" % (dur/3600.))


def test(db, model, opt):
    logging.info("Test started")

    # model setting
    model = set_loaded_model(model, opt=opt)
    model.eval()

    test_generator = db['test']

    test_set = test_generator.dataset
    binc = test_set.binc.to(opt.device)
    dlen = len(test_set)
    test_correct = 0

    probs_placeholder = torch.empty(dlen, opt.ncls).to(opt.device)
    zphot_placeholder = torch.empty(dlen).to(opt.device)
    with torch.no_grad():
        for bepoch, (local_batch, local_zbin) in enumerate(test_generator):
            local_batch = local_batch.to(opt.device)
            local_zbin = local_zbin.to(opt.device)

            # input into model
            out_probs = model(local_batch)

            # get the index of the maximum log-probability
            pred = out_probs.data.max(1, keepdim=True)[1]
            correct_mask = pred.eq(local_zbin.data.view_as(pred))
            test_correct += correct_mask.cpu().sum()

            zphot = torch.sum(out_probs*binc, dim=1).view(-1)

            sidx = bepoch*opt.batch_size
            eidx = sidx+opt.batch_size

            probs_placeholder[sidx:eidx] = out_probs
            zphot_placeholder[sidx:eidx] = zphot

    tdat_len = len(test_generator.dataset)
    test_acc = float(test_correct)/tdat_len
    log_msg = "Test accuracy: %i/%i (%.6f)\n" % \
        (test_correct, tdat_len, test_acc)
    logging.info(log_msg)

    probs = probs_placeholder.cpu().detach().numpy()
    zphot = zphot_placeholder.cpu().detach().numpy()
    outputs = np.hstack((probs, zphot.reshape(-1, 1)))
    save_results(outputs, opt)


def infer(db, model, opt):
    logging.info("Inference started")

    # model setting
    model = set_loaded_model(model, opt=opt)
    model.eval()

    test_generator = db['infer']

    test_set = test_generator.dataset
    binc = test_set.binc.to(opt.device)
    dlen = len(test_set)

    probs_placeholder = torch.empty(dlen, opt.ncls).to(opt.device)
    zphot_placeholder = torch.empty(dlen).to(opt.device)
    zmode_placeholder = torch.empty(dlen).to(opt.device)
    zsig_placeholder = torch.empty(dlen).to(opt.device)
    with torch.no_grad():
        for bepoch, (local_batch, local_zbin) in enumerate(test_generator):
            local_batch = local_batch.to(opt.device)
            local_zbin = local_zbin.to(opt.device)

            # input into model
            out_probs = model(local_batch)

            # get the index of the maximum log-probability
            pred = out_probs.data.max(1, keepdim=True)[1]

            # average redshifts
            zphot = torch.sum(out_probs*binc, dim=1).view(-1)

            # mode redshifts
            prob_argmax = torch.argmax(out_probs, dim=1)
            zmode = binc[prob_argmax]

            # standard deviation
            zsig = torch.sum(out_probs*(binc-zphot.view(-1, 1))**2., dim=1)

            sidx = bepoch*opt.batch_size
            eidx = sidx+opt.batch_size

            probs_placeholder[sidx:eidx] = out_probs
            zphot_placeholder[sidx:eidx] = zphot
            zmode_placeholder[sidx:eidx] = zmode
            zsig_placeholder[sidx:eidx] = zsig

    probs = probs_placeholder.cpu().detach().numpy()
    zphot = zphot_placeholder.cpu().detach().numpy()
    zmode = zmode_placeholder.cpu().detach().numpy()
    zsig = zsig_placeholder.cpu().detach().numpy()
    outputs = np.hstack((probs,
                         zphot.reshape(-1, 1),
                         zmode.reshape(-1, 1),
                         zsig.reshape(-1, 1)))
    for ind in range(0, len(zphot)):
        print("Photo-z: %d %.6f %.6f %.6f" % (ind+1, zphot[ind], zmode[ind], zsig[ind]))
    save_results(outputs, opt)


def save_results(outputs, opt):
    if not os.path.exists(opt.out_fd):
        os.makedirs(opt.out_fd)
    if opt.infer:
        fn = 'inference_output.npy'
    elif opt.test:
        fn = 'test_output.npy'
    out_fn = os.path.join(opt.out_fd, fn)

    np.save(out_fn, outputs)
    logging.info("Outputs are saved at %s" % out_fn)


def set_loaded_model(model, optim=None, opt=None):
    resume_checkpoint = Checkpoint.load(
        model, optim=optim, opt=opt)
    model = resume_checkpoint.model
    model.to(opt.device)

    if optim:
        optim = resume_checkpoint.optim
        return model, optim
    else:
        return model


def main():
    opt = utils.Parser()

    if torch.cuda.is_available():
        opt.device = torch.device('cuda:%s' % opt.gpuid)
    else:
        logging.warning("RUN WITHOUT GPU")
        opt.device = torch.device('cpu')

    db = prepare_db(opt)
    opt = prepare_loss(opt)
    model = prepare_model(opt)

    if opt.train:
        optim = prepare_optim(model)
        train(db, model, optim, opt)
    elif opt.infer:
        infer(db, model, opt)
    else:
        test(db, model, opt)


if __name__ == '__main__':
    main()
