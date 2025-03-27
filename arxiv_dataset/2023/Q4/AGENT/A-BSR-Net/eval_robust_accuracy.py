import os
import time
import pickle
import argparse
import tensorflow as tf
import foolbox as fb
from foolbox import TensorFlowModel, Model
from utils import *
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Robustness evaluations of Keras models via Foolbox')
    parser.add_argument('--data', type=str, default="cifar10", help='dataset for classification')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in dataset')
    parser.add_argument('--model', type=str, default="vgg16", help='network architecture')
    parser.add_argument('--objective', type=str, default="at", help='losses: natural, at, mat, trades, mart, rst')
    parser.add_argument('--pgd_epsilon', type=float, default=8/255, help='perturbation budget epsilon for PGD attacks')
    parser.add_argument('--pgd_iters', type=int, default=50, help='number of PGD steps during attacks')
    parser.add_argument('--pgd_restarts', type=int, default=10, help='number of PGD attack restarts')
    parser.add_argument('--batch', '-b', type=int, default=128, help='training mini-batch size')
    parser.add_argument('--sparse_train', '-s', action='store_true', help='perform robust end-to-end sparse training')
    parser.add_argument('--connectivity', '-pc', type=float, default=0.01, help='sparse connectivity (default: 1%)')
    args = parser.parse_args()

    # Load model and data components
    _, (x_test, y_test), _ = data_loader(args)
    filename = get_filename(args)
    test_batch_size = args.batch
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch_size)

    e_seq = list(np.arange(9, 200, 10))
    rob_acc_seq = []
    for e in e_seq:
        # Load the weights saved by run_connectivity_sampling.py and load into the Keras implementation of models
        w_dict = pickle.load(open(os.path.join('results_new', filename + '_epoch' + str(e) + '_weights.pickle'), 'rb'))
        keras_model = keras_model_loader(args, w_dict['param_list'])
        keras_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

        # Clean test accuracy evaluation
        #keras_model.evaluate(x_test, y_test, batch_size=test_batch_size)

        # White box PGD Attack using the Keras model equivalent
        fmodel: Model = TensorFlowModel(keras_model, bounds=(0, 1))
        print('model')
        pgd_stepsize = (2.5 * args.pgd_epsilon) / args.pgd_iters
        pgd_success = []
        for (images, labels) in test_ds:
            t1 = time.time()
            pgd_attack = fb.attacks.LinfPGD(steps=args.pgd_iters, abs_stepsize=pgd_stepsize, random_start=True)
            t2 = time.time()
            #print('time for fb.attacks.LinfPGD: {:.3g}s'.format(t2-t1))
            fooled = []
            for idx in range(args.pgd_restarts):
                t2_1 = time.time()
                _, _, pgd_is_adv = pgd_attack(fmodel, images, criterion=fb.criteria.Misclassification(labels),
                                            epsilons=args.pgd_epsilon)
                fooled.append(pgd_is_adv)
                t2_2 = time.time()
                #print('ite: {} time: {:.3g}s'.format(idx, t2_2-t2_1))
            pgd_is_adv = tf.reduce_any(fooled, axis=0)
            pgd_success.append(sum(pgd_is_adv.numpy()))
            t3 = time.time()
            print('time: {:.3g}s robust acc: {:.3g}'.
                    format(t3-t1, 1-sum(pgd_success)/args.batch))
        acc_e = 1 - (sum(pgd_success) / x_test.shape[0])
        print('PGD robust accuracy: ' + str(acc_e))
        rob_acc_seq.append(acc_e)
    print(rob_acc_seq)
    name_out = 'rob_acc_' + args.objective + '_' + str(int(args.connectivity*100)) + '_' + \
               args.data + '_' + args.model +  '.csv'
    np.savetxt(name_out, rob_acc_seq, delimiter=',')


if __name__ == '__main__':
    main()
