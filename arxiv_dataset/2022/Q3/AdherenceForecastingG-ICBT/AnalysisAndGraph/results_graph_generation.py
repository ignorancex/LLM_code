import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score


def plotpoint_results(results_variable_length):
    dict_for_dataframe = {"Sequence Length (day)": [], "Current Repetition": [], "Balanced Accuracy (%)": []}
    for entry in results_variable_length:
        dict_for_dataframe["Sequence Length (day)"].append(entry["Sequence Length"])
        dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
        dict_for_dataframe["Balanced Accuracy (%)"].append(int(entry["Balanced Accuracy"] * 100.))
    df = pd.DataFrame.from_dict(dict_for_dataframe)
    print(df)
    sns.set(font_scale=3.7)
    sns.set_style("ticks")
    g = sns.pointplot(data=df, x="Sequence Length (day)", y="Balanced Accuracy (%)", err_style="bars", marker="o",
                      ci=95,
                      capsize=.5, scale=2.5, errwidth=7, linestyles="-", linewidth=20)
    palette = itertools.cycle(sns.color_palette())
    next(palette)
    color_threshold = next(palette)
    plt.axhline(50, ls="--", color=next(palette), linewidth=8, label="Threshold 1")
    plt.axhline(65, ls="-.", color=next(palette), linewidth=8, label="Threshold 2")
    plt.axhline(70, ls=":", color=next(palette), linewidth=8, label="Threshold 3")
    handles, _ = g.get_legend_handles_labels()
    plt.legend()
    sns.despine(offset=1)


def confusion_matrix(results, sequence_length_to_consider):
    confusion_matrix_to_show = None
    number_of_instance_confusion_matrix = 0
    balanced_accuracy = 0.
    for entry in results:
        if sequence_length_to_consider == entry["Sequence Length"]:
            number_of_instance_confusion_matrix += 1
            if confusion_matrix_to_show is None:
                confusion_matrix_to_show = entry["Confusion Matrix"]
            else:
                confusion_matrix_to_show += entry["Confusion Matrix"]
            balanced_accuracy += entry["Balanced Accuracy"] * 100.
    balanced_accuracy /= number_of_instance_confusion_matrix
    print(confusion_matrix_to_show / number_of_instance_confusion_matrix)
    cm = ConfusionMatrixDisplay(
        confusion_matrix=np.array(confusion_matrix_to_show / number_of_instance_confusion_matrix),
        display_labels=["Dropout", "Adherent"])
    sns.set(font_scale=3.7)
    sns.set_style("ticks")
    cm.plot(values_format='', colorbar=False)
    plt.title("Sequence length: {} \nBalanced Accuracy: {:.2f}%".format(sequence_length_to_consider, balanced_accuracy))

    # plt.figure()


def get_statistical_values_fixed_lenght_patient_different_dataset(results_variable_length,
                                                                  results_differents_single_day, sequence_lengths):
    print("RESULTS VARIABLE LENGTH: \n\n")
    for sequence_length in sequence_lengths:
        ground_truths, predictions = [], []
        for entry in results_variable_length:
            if entry["Sequence Length"] == sequence_length:
                # print(entry["ground_truths"])
                # print(entry["predictions"])
                # print(entry["Current Repetition"])
                ground_truths.append(entry["ground_truths"])
                predictions.append(entry["predictions"])
        ground_truths = np.array(ground_truths).T
        predictions = np.array(predictions).T
        accuracies_score = []
        for ground_truth_current_repetition, prediction_current_repetition in zip(ground_truths, predictions):
            accuracy = balanced_accuracy_score(prediction_current_repetition, ground_truth_current_repetition)
            accuracies_score.append(accuracy)
        print("Sequence Length: ", sequence_length, " Accuracy score: \n", accuracies_score)
        print("Average accuracy: ", np.mean(accuracies_score))

        ground_truths = np.array(ground_truths).T
        predictions = np.array(predictions).T
        accuracies_score = []
        for ground_truth_current_repetition, prediction_current_repetition in zip(ground_truths, predictions):
            accuracy = balanced_accuracy_score(prediction_current_repetition, ground_truth_current_repetition)
            accuracies_score.append(accuracy)
        print("Sequence Length: ", sequence_length, " Accuracy score: \n", accuracies_score)
        print("Average accuracy: ", np.mean(accuracies_score))

    print("RESULTS FIXED LENGTH: \n\n")
    sequence_length = [7, 11, 20, 42]
    for i, _ in enumerate(results_differents_single_day):
        ground_truths, predictions = [], []
        for entry in results_differents_single_day[i]:
            # print(entry)
            # print(entry["ground_truths"])
            # print(entry["predictions"])
            # print(entry["Current Repetition"])
            ground_truths.append(entry["ground_truths"])
            predictions.append(entry["predictions"])
        ground_truths = np.array(ground_truths).T
        predictions = np.array(predictions).T
        accuracies_score = []
        for ground_truth_current_repetition, prediction_current_repetition in zip(ground_truths, predictions):
            accuracy = balanced_accuracy_score(prediction_current_repetition, ground_truth_current_repetition)
            accuracies_score.append(accuracy)
        print("Sequence Length: ", sequence_length[i], " Accuracy score: \n", accuracies_score)
        print("Average accuracy: ", np.mean(accuracies_score))


def get_statistical_information_multisequence_independent_runs(results_variable_length, results_differents_single_day,
                                                               sequence_lengths):
    dict_for_dataframe = {"Sequence Length (day)": [], "Current Repetition": [], "Balanced Accuracy (%)": [],
                          "Training type": []}
    for entry in results_variable_length:
        if entry["Sequence Length"] in sequence_lengths:
            dict_for_dataframe["Sequence Length (day)"].append(entry["Sequence Length"])
            dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
            dict_for_dataframe["Balanced Accuracy (%)"].append(entry["Balanced Accuracy"] * 100.)
            dict_for_dataframe["Training type"].append("All Sequence Lengths Simultaneously")
    for i, lenght in enumerate(sequence_lengths):
        print(lenght)
        for entry in results_differents_single_day[i]:
            dict_for_dataframe["Sequence Length (day)"].append(lenght)
            dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
            dict_for_dataframe["Balanced Accuracy (%)"].append(entry["Balanced Accuracy"] * 100.)
            dict_for_dataframe["Training type"].append("Fixed Sequence Length")
    df = pd.DataFrame.from_dict(dict_for_dataframe)

    for sequence_length in sequence_lengths:
        balanced_accuracy_multi_sequence = df["Balanced Accuracy (%)"].loc[
            (df["Training type"] == "All Sequence Lengths Simultaneously") & (
                    df["Sequence Length (day)"] == sequence_length)]
        print("Multi sequence length. Length size: ", sequence_length)
        print(balanced_accuracy_multi_sequence.values.tolist())

        balanced_accuracy_fixed_sequence = df["Balanced Accuracy (%)"].loc[
            (df["Training type"] == "Fixed Sequence Length") & (df["Sequence Length (day)"] == sequence_length)]
        print("Fixed sequence length. Length size: ", sequence_length)
        print(balanced_accuracy_fixed_sequence.values.tolist())
        print("\n\n")


def histogram_comparison_multisequence_vs_single_day(results_variable_length, results_differents_single_day,
                                                     sequence_lengths):
    dict_for_dataframe = {"Sequence Length (day)": [], "Current Repetition": [], "Balanced Accuracy (%)": [],
                          "Training type": []}
    for entry in results_variable_length:
        if entry["Sequence Length"] in sequence_lengths:
            dict_for_dataframe["Sequence Length (day)"].append(entry["Sequence Length"])
            dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
            dict_for_dataframe["Balanced Accuracy (%)"].append(entry["Balanced Accuracy"] * 100.)
            dict_for_dataframe["Training type"].append("All Sequence Lengths Simultaneously")
    for i, lenght in enumerate(sequence_lengths):
        print(lenght)
        for entry in results_differents_single_day[i]:
            dict_for_dataframe["Sequence Length (day)"].append(lenght)
            dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
            dict_for_dataframe["Balanced Accuracy (%)"].append(entry["Balanced Accuracy"] * 100.)
            dict_for_dataframe["Training type"].append("Fixed Sequence Length")
    df = pd.DataFrame.from_dict(dict_for_dataframe)

    print(df["Balanced Accuracy (%)"].loc[
              (df["Training type"] == "Fixed Sequence Length") & (df["Sequence Length (day)"] == 7)])
    sns.set(font_scale=3.7)
    sns.set_style("ticks")
    ax = sns.barplot(data=df, x="Sequence Length (day)", y="Balanced Accuracy (%)", hue="Training type", ci=95,
                     capsize=.2, errwidth=7)

    hatches = ["", "/"]
    # Loop over the bars
    for bars, hatch in zip(ax.containers, hatches):
        # Set a different hatch for each group of bars
        for bar in bars:
            bar.set_hatch(hatch)

    palette = itertools.cycle(sns.color_palette())
    next(palette)
    next(palette)
    plt.axhline(50, ls="--", color=next(palette), linewidth=8, label="_nolegend_", zorder=10)
    plt.axhline(65, ls="-.", color=next(palette), linewidth=8, label="_nolegend_", zorder=10)
    plt.axhline(70, ls=":", color=next(palette), linewidth=8, label="_nolegend_", zorder=10)
    sns.despine()
    ax.legend(title="Training type")
    plt.show()


def plot_points_performances_weighthings(results_variable_length, results_variable_length_no_weight):
    dict_for_dataframe = {"Sequence Length (day)": [], "Current Repetition": [], "Balanced Accuracy (%)": [],
                          "Weighting Scheme": []}
    for entry in results_variable_length:
        dict_for_dataframe["Sequence Length (day)"].append(entry["Sequence Length"])
        dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
        dict_for_dataframe["Balanced Accuracy (%)"].append(int(entry["Balanced Accuracy"] * 100.))
        dict_for_dataframe["Weighting Scheme"].append("Per-Class Weighting")
    for entry in results_variable_length_no_weight:
        dict_for_dataframe["Sequence Length (day)"].append(entry["Sequence Length"])
        dict_for_dataframe["Current Repetition"].append(entry["Current Repetition"])
        dict_for_dataframe["Balanced Accuracy (%)"].append(int(entry["Balanced Accuracy"] * 100.))
        dict_for_dataframe["Weighting Scheme"].append("No Weighting")
    df = pd.DataFrame.from_dict(dict_for_dataframe)
    print(df)

    sns.set(font_scale=3.7)
    sns.set_style("ticks")
    palette = itertools.cycle(sns.color_palette())
    next(palette)
    next(palette)

    plt.axhline(50, ls="--", color=next(palette), linewidth=8, label="_nolegend_", zorder=0)
    plt.axhline(65, ls="-.", color=next(palette), linewidth=8, label="_nolegend_", zorder=0)
    plt.axhline(70, ls=":", color=next(palette), linewidth=8, label="_nolegend_", zorder=0)
    g = sns.pointplot(data=df, x="Sequence Length (day)", y="Balanced Accuracy (%)", err_style="bars", marker="o",
                      ci=95, capsize=.5, scale=2.5, errwidth=7, linestyles="-", linewidth=20, hue="Weighting Scheme")
    g.lines[3].set_linestyle(":")
    plt.legend()
    sns.despine(offset=1)
    plt.show()


if __name__ == '__main__':
    path_results_self_attention_variable_length_all_weigthing = "../Results/56_days_batch_64_var_len_self_attention_all_weigthing.txt"
    with open(path_results_self_attention_variable_length_all_weigthing, "rb") as file:  # Unpickling
        results_variable_length_all_weigthing = pickle.load(file)

    path_results_self_attention_per_class_weigthing = "../Results/56_days_batch_64_var_len_self_attention_per_class_weigth.txt"
    with open(path_results_self_attention_per_class_weigthing, "rb") as file:  # Unpickling
        results_variable_length_per_class_weigthing = pickle.load(file)
    path_results_self_attention_variable_length_per_sample_weigthing = "../Results/56_days_batch_64_var_len_self_attention_per_sample_weight.txt"
    with open(path_results_self_attention_variable_length_per_sample_weigthing, "rb") as file:  # Unpickling
        results_variable_length_per_sample_weigthing = pickle.load(file)

    path_results_self_attention_variable_length_no_weigthing = "../Results/56_days_batch_64_var_len_self_attention_no_weight.txt"
    with open(path_results_self_attention_variable_length_no_weigthing, "rb") as file:  # Unpickling
        results_variable_length_no_weigthing = pickle.load(file)

    print(results_variable_length_per_class_weigthing)
    print(np.shape(results_variable_length_per_class_weigthing))

    sequence_lengths = [7, 11, 20, 42]

    plotpoint_results(results_variable_length_per_class_weigthing)

    for length in sequence_lengths:
        confusion_matrix(results=results_variable_length_per_class_weigthing, sequence_length_to_consider=length)
    plt.show()

    results_differents_single_day = []
    for length in sequence_lengths:
        path_results = "../Results/self_attention_test_performance_length_%d.txt" % length
        with open(path_results, "rb") as file:  # Unpickling
            results_differents_single_day.append(pickle.load(file))

    histogram_comparison_multisequence_vs_single_day(
        results_variable_length=results_variable_length_per_class_weigthing,
        results_differents_single_day=results_differents_single_day,
        sequence_lengths=sequence_lengths)

    plot_points_performances_weighthings(results_variable_length_per_class_weigthing,
                                         results_variable_length_no_weigthing)

    get_statistical_information_multisequence_independent_runs(
        results_variable_length=results_variable_length_per_class_weigthing,
        results_differents_single_day=results_differents_single_day, sequence_lengths=sequence_lengths)
