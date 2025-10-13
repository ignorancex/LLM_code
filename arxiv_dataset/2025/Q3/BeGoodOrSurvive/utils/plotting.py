import matplotlib.pyplot as plt

def plot_loss_and_survival(average_loss_per_gen, survival_counts, filename='loss_survival_plot.png'):
    """
    Plot Loss and Survival on the same plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Average Loss
    ax1.plot(range(1, len(average_loss_per_gen) + 1), average_loss_per_gen, label='Average Loss per Generation', marker='o', color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot Survival Counts
    ax1.plot(range(1, len(survival_counts) + 1), survival_counts, label='Rounds Survived per Game', marker='x', color='orange')

    # Add a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Progress of Survival and Loss Across Generations')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")


def plot_survival_and_ethics(survival_counts, average_ethical_score_per_gen, filename='survival_ethics_plot.png'):
    """
    Plot Survival and Ethics on the same plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Survival Counts
    ax1.plot(range(1, len(survival_counts) + 1), survival_counts, label='Rounds Survived per Game', marker='x', color='orange')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Survival', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a secondary y-axis for Ethical Scores
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(average_ethical_score_per_gen) + 1), average_ethical_score_per_gen, label='Average Ethical Score per Generation', marker='^', color='green')
    ax2.set_ylabel('Ethical Score', color='green')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Progress of Survival and Ethical Scores Across Generations')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")


def plot_loss_and_ethics(average_loss_per_gen, average_ethical_score_per_gen, filename='loss_ethics_plot.png'):
    """
    Plot Loss and Ethics on the same plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Average Loss
    ax1.plot(range(1, len(average_loss_per_gen) + 1), average_loss_per_gen, label='Average Loss per Generation', marker='o', color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for Ethical Scores
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(average_ethical_score_per_gen) + 1), average_ethical_score_per_gen, label='Average Ethical Score per Generation', marker='^', color='green')
    ax2.set_ylabel('Ethical Score', color='green')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Progress of Loss and Ethical Scores Across Generations')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")

