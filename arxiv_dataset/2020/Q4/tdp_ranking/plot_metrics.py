import argparse
import re

import matplotlib.pyplot as plt


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log_file", help="log file to plot")
    arg_parser.add_argument("--output_dir", help="output directory for plotted image")

    return arg_parser


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    lines = open(args.log_file, 'r').readlines()

    metric_line_regex = r' - \d+s - loss: [\d\.]+ - .*accuracy: [\d\.]+ - val_loss: [\d\.]+ - val_.*accuracy: [\d\.]+'
    epoch_lines = [line for line in lines if re.match(metric_line_regex, line)]

    training_losses = []
    dev_losses = []
    training_accuracies = []
    dev_accuracies = []

    for i, epoch_line in enumerate(epoch_lines):
        # Sections:
        # 0 - progress bar (if it exists)
        # 1 - total time & time/step
        # 2 - training loss
        # 3 - training accuracy
        # 4 - dev/validation loss
        # 5 - dev/validation accuracy
        sections = epoch_line.split(' - ')

        training_losses.append(float(sections[2].split(': ')[1]))
        training_accuracies.append(float(sections[3].split(': ')[1]))
        dev_losses.append(float(sections[4].split(': ')[1]))
        dev_accuracies.append(float(sections[5].split(': ')[1]))

    epochs = range(len(training_losses))

    plt.figure()
    loss_plot = plt.subplot(2, 1, 1)
    loss_plot.plot(epochs, training_losses, c='b', label='training')
    loss_plot.plot(epochs, dev_losses, c='r', label='dev')
    plt.xlabel('Iteration #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    acc_plot = plt.subplot(2, 1, 2)
    acc_plot.plot(epochs, training_accuracies, c='b', label='training')
    acc_plot.plot(epochs, dev_accuracies, c='g', label='dev')
    plt.xlabel('Iteration #')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    exp_id_pieces = args.log_file.split('/')[-1].split('-')
    exp_id = exp_id_pieces[0]
    exp_run_id = exp_id_pieces[2].split('.')[0]
    save_file = exp_id + '-' + exp_run_id + '.png'  # e.g. '46572_bert_classifier-0.png'
    plt.savefig(args.output_dir + '/' + save_file)

    plt.show()
