import argparse
import os
import sys

from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer.datasets import TransformDataset

from SSD_for_vehicle_detection import *
from DA_updater import *
from COWC_dataset_processed import Dataset_imgonly, COWC_dataset_processed, vehicle_classes
from SSD_training import  Transform
from utils import initSSD
from SSD_test import ssd_evaluator
import chainermn

sys.path.append(os.path.dirname(__file__))

def make_optimizer(model, alpha, beta1, beta2,comm):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    if comm:
        optimizer =  chainermn.create_multi_node_optimizer(
            optimizer, comm)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--snapshot_iterations', type=int, nargs="*")
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/addatest', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10, help='Interval of evaluation')
    parser.add_argument('--evaluation_interval_s', type=int, help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--dis_adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--gen_adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')
    parser.add_argument('--dis_file', type=str, help='discriminator file path for initialization')
    parser.add_argument('--pred_align_weight', type=float, default=1.)
    parser.add_argument('--pred_dis_class', type=str, default="Cls_discriminator")
    parser.add_argument('--pred_align_mode', type=int, choices=[0,1,2,3,4], default=1,help="0: no normalization, "
                                                                                        "1: weight all classes equally, "
                                                                                        "2: update weight every epoch (align to the Most frequent class), "
                                                                                        "3: update weight every epoch (normalize), "
                                                                                        "4: use fixed weight")
    parser.add_argument('--pred_align_norm_weight', type=float, nargs=2 )
    parser.add_argument('--pred_align_norm_default_weight', type=float, nargs=2,default=[3,1])
    parser.add_argument('--generator_class', type=str, help='class name of generator',default="Generator_VGG16_simple")
    parser.add_argument('--gen_file', type=str)
    parser.add_argument('--source_dataset', type=str, help='source dataset directory')
    parser.add_argument('--target_dataset', type=str, help='target dataset directory')
    parser.add_argument('--ssdpath', type=str,  help='SSD model file')
    parser.add_argument('--evalimg', type=str, help='img path for evaluation')
    parser.add_argument('--evalimg_s', type=str, help='img path for evaluation in source domain')
    parser.add_argument('--evalimg_s_suffix', type=str, default="_s")
    parser.add_argument('--eval_src_save_snapshot', action="store_true")
    parser.add_argument('--resume', type=str, help='trainer snapshot path for resume')
    parser.add_argument('--bufsize', type=int, help='size of buffer for discriminator training')
    parser.add_argument('--bufmode', type=int, default=1, help='mode of buffer(0:align src and tgt, 1:not align, 2:sort by loss value)')
    parser.add_argument('--out_progress')
    parser.add_argument('--snapshot_retain', type=int, default=3)
    parser.add_argument('--cls_batch_split', type = int)
    parser.add_argument('--mboxloss_mode', type=int, choices=[0,1], default=1)
    parser.add_argument('--RTN_mbox_class', type=str, default="Mbox_custom")
    parser.add_argument('--RTN_mbox_ch', type=int, nargs=6)
    parser.add_argument('--mbox_delta_inv', type=int, choices = [0,1,2,3, 4, 5,6,7,8], default=8, help="0: not apply, 1: sub-det only (GRL), 2: sub-det and enc (GRL), 3: sub-det and enc (adv_inv), 4: sub-det only (adv_inv),"
                                                                                    "5: whole det and enc (GRL), 6: whole det only (GRL), 7: whole det and enc (adv_inv), 8: whole det only (adv_inv)")
    parser.add_argument('--mbox_delta_inv_weight', type=float, default=1.)
    parser.add_argument('--communicator', type=str,
                        default='pure_nccl', help='Type of communicator')
    parser.add_argument('--min_entropy_type', type=int,choices=[1,2],default=1,help="1:devide by bsize * dbox number, 2: devide by bsize")
    parser.add_argument('--min_entropy_coef', type=float, default=0.3)
    parser.add_argument('--use_mn', action="store_true")
    parser.add_argument('--method', type=str, choices=["plain_adv","no_norm","norm_D_P","norm_P","emin","coGAN"], default="norm_P")

    args = parser.parse_args()

    pred_align = False
    min_entropy = None
    RTN_mbox = None
    pred_align_mode = 0
    coGAN = False
    disable_pred_align_norm = [False,False,False]
    pred_align_weight = args.pred_align_weight
    pred_align_norm_default_weight = args.pred_align_norm_default_weight
    if args.method in ["no_norm","norm_D_P","norm_P"]:
        pred_align = True
        if args.method == "no_norm":
            pred_align_mode = 0
        elif args.method == "norm_D_P":
            pred_align_mode = args.pred_align_mode
            pred_align_weight = 0.1
            pred_align_norm_default_weight = [1,1]
        elif args.method == "norm_P":
            pred_align_mode = args.pred_align_mode
            disable_pred_align_norm = [True,True,False]
    elif args.method == "emin":
        min_entropy = args.min_entropy_type
    elif args.method == "coGAN":
        coGAN = True

    if not args.use_mn:
        comm = None
        device = args.gpu
    else:
        if args.gpu >= 0:
            if args.communicator == 'naive':
                print('Error: \'naive\' communicator does not support GPU.\n')
                exit(-1)
            comm = chainermn.create_communicator(args.communicator)
            device = comm.intra_rank
        else:
            if args.communicator != 'naive':
                print('Warning: using naive communicator '
                      'because only naive supports CPU-only execution')
            comm = chainermn.create_communicator('naive')
            device = -1

    if comm and comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num iteration: {}'.format(args.max_iter))
        print('==========================================')

    if not comm or comm.rank == 0:
        if not os.path.isdir(args.out):
            os.makedirs(args.out)

    chainer.cuda.get_device_from_id(device).use()

    report_keys = ["loss_cls", "loss_loc", "loss_conf", "loss_t_enc", "loss_dis", 'loss_dis_src', 'loss_dis_tgt',  'lr_dis', 'lr_cls']
    if args.evalimg:
        report_keys += ['validation/main/map',
                   'validation/main/RR/car',
                   'validation/main/PR/car', 'validation/main/FAR/car', 'validation/main/F1/car']
    if args.evalimg_s:
        report_keys += ['validation{}/main/map'.format(args.evalimg_s_suffix),
                   'validation{}/main/RR/car'.format(args.evalimg_s_suffix),
                   'validation{}/main/PR/car'.format(args.evalimg_s_suffix),
                        'validation{}/main/FAR/car'.format(args.evalimg_s_suffix),
                        'validation{}/main/F1/car'.format(args.evalimg_s_suffix)]

    if coGAN:
        report_keys += ["loss_cls_org"]

    if min_entropy:
        report_keys += ['loss_t_entropy']

    if args.method in ["no_norm","norm_D_P","norm_P"]:
        report_keys += ['loss_dis_cls_enc', 'loss_dis_cls', 'loss_dis_cls_src', 'loss_dis_cls_tgt']

    if coGAN:
        discriminator = DA4_discriminator_bn()
    else:
        discriminator = DA4_discriminator()
    if args.dis_file:
        serializers.load_npz(args.dis_file, discriminator)

    ssd_model = initSSD("ssd300",0.3,args.ssdpath)
    models = [discriminator, ssd_model]

    if not comm or comm.rank == 0:
        source_dataset = TransformDataset(
            COWC_dataset_processed(split="train", datadir=args.source_dataset),
            Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
        target_dataset_ = Dataset_imgonly(args.target_dataset)
        target_dataset = TransformDataset(target_dataset_,
                Transform(ssd_model.coder, ssd_model.insize, ssd_model.mean))
    else:
        source_dataset = None
        target_dataset = None

    if comm:
        source_dataset = chainermn.scatter_dataset(source_dataset, comm, shuffle=True)
        target_dataset = chainermn.scatter_dataset(target_dataset, comm, shuffle=True)

    train_iter1 = chainer.iterators.MultiprocessIterator(source_dataset, args.batchsize)
    train_iter2 = chainer.iterators.MultiprocessIterator(target_dataset, args.batchsize)

    updater_args = {
        "iterator": {'main': train_iter1, 'target': train_iter2, },
        "device": device,
        "comm":comm,
    }

    updater_args["RTN_mbox"] = RTN_mbox
    updater_args["mbox_delta_inv"] = args.mbox_delta_inv
    updater_args["mbox_delta_inv_weight"] = args.mbox_delta_inv_weight
    updater_args["min_entropy"] = min_entropy
    updater_args["min_entropy_coef"] = args.min_entropy_coef
    updater_args["pred_align"] = pred_align
    updater_args["pred_align_weight"] = pred_align_weight
    updater_args["pred_align_mode"] = pred_align_mode
    updater_args["pred_align_norm_default_weight"] = pred_align_norm_default_weight
    if pred_align_mode == 4:
        updater_args["pred_align_norm_weight"] = args.pred_align_norm_weight

    updater_args["disable_pred_align_norm"] = disable_pred_align_norm

    if args.bufsize < int(args.batchsize/2):
        print("bufsize must not be smaller than batchsize/2")
        raise ValueError
    buffer = fmapBuffer(args.bufsize,mode=args.bufmode,discriminator=discriminator,gpu=device)
    updater_args["buffer"] = buffer

    updater_args["cls_batch_split"] = args.cls_batch_split if args.cls_batch_split else args.batchsize
    updater_args["mboxloss_mode"] = args.mboxloss_mode

    if coGAN:
        gen_args = {}
        gen_args["norm"] = "bn"
        gen_args["comm"] = comm
        gen_args["residual"] = True
        generator = eval(args.generator_class)(**gen_args)
        if args.gen_file:
            serializers.load_npz(args.gen_file, generator)
        updater_args["generator"] = generator
    else:
        updater_args["generator"] = None
    updater_args["coGAN"] = coGAN

    # Set up optimizers
    opts = {}
    opts["opt_dis"] = make_optimizer(discriminator, args.adam_alpha, args.adam_beta1, args.adam_beta2,comm=comm)
    opts["opt_cls"] = make_optimizer(ssd_model, args.adam_alpha, args.adam_beta1, args.adam_beta2,comm=comm)
    if coGAN:
        opts["opt_gen"] = make_optimizer(generator, args.gen_adam_alpha, args.adam_beta1, args.adam_beta2,comm=comm)
    if RTN_mbox:
        mbox_delta = eval(args.RTN_mbox_class)(len(vehicle_classes) + 1, ssd_model.multibox.aspect_ratios,ch_num=args.RTN_mbox_ch)
        opts["opt_mbox_delta"] = make_optimizer(mbox_delta, args.adam_alpha, args.adam_beta1, args.adam_beta2,
                                              comm=comm)
    if pred_align:
        cls_dis = eval(args.pred_dis_class)()
        opts["opt_cls_dis"] = make_optimizer(cls_dis, args.adam_alpha, args.adam_beta1, args.adam_beta2,
                                              comm=comm)

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    updater_args["snapshot_interval"] = args.snapshot_interval
    updater_args["outdir"] = args.out
    updater_args["adv_inv"] = True if coGAN else False

    updater = Adv_updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    if not comm or comm.rank == 0:

        trainer.extend(extensions.observe_lr(optimizer_name="opt_cls", observation_key='lr_cls'),
                       trigger=(args.display_interval, 'iteration'))
        trainer.extend(extensions.snapshot(num_retain=args.snapshot_retain), trigger=(args.snapshot_interval, 'iteration'))
        trainer.extend(extensions.LogReport(keys=report_keys,
                                            trigger=(args.display_interval, 'iteration')))

        bestshot_dir = os.path.join(args.out,"bestshot")
        if args.evalimg and not os.path.isdir(bestshot_dir): os.makedirs(bestshot_dir)
        if args.evalimg_s and not os.path.isdir(bestshot_dir+args.evalimg_s_suffix): os.makedirs(bestshot_dir+args.evalimg_s_suffix)

        printreport_args = {"entries": report_keys}
        progress_args = {"update_interval": 10}
        if args.out_progress:
            fo = open(args.out_progress, 'w')
            printreport_args["out"] = fo
            progress_args["out"] = fo

        trainer.extend(extensions.PrintReport(**printreport_args),
                       trigger=(args.display_interval, 'iteration'))
        trainer.extend(extensions.ProgressBar(**progress_args))

        if args.snapshot_iterations:
            trainer.extend(
                extensions.snapshot_object(ssd_model, 'ssdmodel_iter_{.updater.iteration}'),
                trigger=chainer.training.triggers.ManualScheduleTrigger(args.snapshot_iterations, 'iteration'))

        trainer.extend(
            extensions.snapshot_object(ssd_model, 'ssdmodel_iter_{.updater.iteration}'),
            trigger=(args.max_iter, 'iteration'))

        if coGAN:
            trainer.extend(
                extensions.snapshot_object(generator, 'generator_iter_{.updater.iteration}'),
                trigger=(args.max_iter, 'iteration'))

        trainer.extend(
            extensions.snapshot_object(discriminator, 'adv_discriminator_iter_{.updater.iteration}'),
            trigger=(args.max_iter, 'iteration'))

        if RTN_mbox:
            trainer.extend(
                extensions.snapshot_object(mbox_delta, 'mbox_delta_iter_{.updater.iteration}'),
                trigger=(args.max_iter, 'iteration'))

        if args.evalimg:
            trainer.extend(
                ssd_evaluator(
                    args.evalimg, ssd_model,updater,savedir=bestshot_dir, label_names=vehicle_classes),
                trigger=(args.evaluation_interval, 'iteration'))
        else:
            print("No validation in target domain will be executed.")

        if args.evalimg_s:
            trainer.extend(
                ssd_evaluator(
                    args.evalimg_s, ssd_model,updater,savedir=bestshot_dir+args.evalimg_s_suffix, label_names=vehicle_classes, suffix=args.evalimg_s_suffix,save_snapshot=args.eval_src_save_snapshot),
                trigger=(args.evaluation_interval_s if args.evaluation_interval_s else args.evaluation_interval, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    if args.out_progress:
        fo.close()

if __name__ == '__main__':
    main()