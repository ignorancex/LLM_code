import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import X_AXIS
from root.experiments.MNIST.capacity.results import LOG2_THRESHOLDS_SIX_EIGHT, LOG2_THRESHOLDS_ZERO_ONE, RANDOM_DATA_LOG2_TRESHOLDS, MEM_CAPS_0and1, MEM_CAPS_6and8


def plot_progression(title, x_data, y_data, x_label):

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    
    for i in range(len(y_data)):
        ax.plot(x_data, y_data[i], label='mem cap' + str(i))

    ax.set_ylim(0, 30)
    ax.legend(loc='upper left')
    fig.suptitle(title)

    plt.show()

if __name__ == "__main__":
    plot_progression("Memorization capacity and number of thresholds", X_AXIS, [MEM_CAPS_0and1, MEM_CAPS_6and8, LOG2_THRESHOLDS_ZERO_ONE, LOG2_THRESHOLDS_SIX_EIGHT, RANDOM_DATA_LOG2_TRESHOLDS], "Memorization Capacity and Number of Thresholds")