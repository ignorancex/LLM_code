import gc
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import psutil
import torch
from huggingface_hub import login
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class case_study:

    def __init__(self, root_path):
        self.root_path = root_path
        self.stats_files = []

    def find_hallucinations_stats_files(self, root_path, alternative_name='hallucinations_stats_half.json',
                                        factuality=False):
        # List to store all found file paths
        stats_files = []

        # Walk through all directories and files in the given root path
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                # Check if the filename matches 'hallucinations_stats.json'
                part_of_name = alternative_name
                if factuality:
                    part_of_name = 'factuality_stats.json'
                if filename == part_of_name:
                    file_path = os.path.join(dirpath, filename)

                    stats_files.append(file_path)

        return stats_files

    def open_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        # if the data is a list of dictionaries, return it else return None
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            random.seed(42)
            random.shuffle(data)
            return data
        else:
            return None

    def remove_half_hallucinations(self, root_path):
        """
        Remove the hallucinations that might not be hallucinations. Save the new files with the name hallucinations_stats_half.json
        :param root_path:
        :return:
        """
        stats_files = self.find_hallucinations_stats_files(root_path, alternative_name='hallucinations_stats.json')
        import nltk
        # # nltk.download('stopwords')
        # nltk.download('wordnet')
        stop_words = list(set(stopwords.words('english'))) + ["the"]
        for file_path in stats_files:
            data = self.open_file(file_path)
            print(f"File: {file_path}")
            if data is None or len(data) == 0:
                print(f"File: {file_path} is empty")
                continue

            new_file_path = file_path.replace('hallucinations_stats.json', 'hallucinations_stats_half.json')
            new_data = []
            for example in data:
                true_answer = example["true_answer"].lower().strip().replace("-", " ")
                generated = example["generated"].lower().strip()
                model_name = file_path.split("/")[1]
                if model_name == "google_gemma-2-9b-it" and generated.count("*") < 4:
                    continue
                if "the answer is not " in generated:
                    continue
                # print(f"{generated=}")
                generated_answer = \
                generated.split("\nanswer:")[2].strip().replace("assistant\n\n", "").replace("model\nthe answer is",
                                                                                             "").replace(
                    "the answer is", "").split("\n")[0].split(".")[0].split(",")[0].strip().lower().replace("-", " ")
                generated_answer = generated_answer.split(". ")[0]
                # print(f"{true_answer=} {generated_answer=}")
                # remove from both answers "the"
                true_answer = " ".join([word for word in true_answer.split() if word.lower() not in stop_words])
                generated_answer = " ".join(
                    [word for word in generated_answer.split() if word.lower() not in stop_words])

                # check that a synonym of the true answer is not in the generated answer
                synonims = nltk.corpus.wordnet.synsets(true_answer)
                is_syn = False
                for syn in synonims:
                    for l in syn.lemmas():
                        if l.name().replace("_", " ").lower() in generated_answer:
                            is_syn = True
                            break
                if is_syn:
                    continue
                # stem the words
                true_answer = " ".join([nltk.PorterStemmer().stem(word) for word in true_answer.split()])
                generated_answer = " ".join([nltk.PorterStemmer().stem(word) for word in generated_answer.split()])
                dist = nltk.edit_distance(true_answer, generated_answer)
                if len(generated_answer) == 0 or len(true_answer) == 0 or sum(
                        [1 for word in true_answer.split() if word in generated_answer.split()]) >= 0.5 * len(
                        true_answer.split()) \
                        or true_answer.split()[-1].lower() in generated_answer.lower().split():
                    continue

                if (dist > 2 or true_answer.isdigit()) and (
                        len(generated_answer) > 0 and "great" not in generated_answer and "none " not in generated_answer and "n/a" not in generated_answer \
                        and not (
                        generated_answer.split()[0] == true_answer.split()[0] and len(generated_answer.split()) == 1)):
                    new_data.append(example)
            print(f"len data: {len(data)} {len(new_data)=} {round(100 * (len(data) - len(new_data)) / len(data), 2)}%")

            with open(new_file_path, 'w') as file:
                json.dump(new_data, file)

    def mitigation_uncertainty_based(self, root_path):
        """

        :param root_path:
        :return:
        """
        stats_files = self.find_hallucinations_stats_files(root_path)

        mitigations = ["temp_generations", "mean_entropy", "prob"]
        results = {}

        for file_path in stats_files:
            data_hall = self.open_file(file_path)
            data_fact = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))
            if data_hall is None or data_fact is None or "child2" in file_path or "27" in file_path:
                continue
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            settings = file_path.split("/")[3]
            if settings not in results.keys():
                results[settings] = {}
                results[settings] = {}

            if dataset not in results[settings].keys():
                results[settings][dataset] = {}
                results[settings][dataset] = {}

            if model_name not in results[settings][dataset].keys():
                results[settings][dataset][model_name] = {}
                results[settings][dataset][model_name] = {}
            results[settings][dataset][model_name]["temp_generations"] = {}
            results[settings][dataset][model_name]["mean_entropy"] = {}
            results[settings][dataset][model_name]["prob"] = {}

            print(f"File: {file_path}")
            for mitigation in mitigations:
                print(f"Mitigation: {mitigation}")
                threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss = self.get_threshold(
                    data_hall,
                    data_fact,
                    mitigation)
                print(f"{mitigation=} {model_name=} {dataset=} {threshold=}")
                if "temp_generations" in mitigation:
                    number_of_not_mitigated = sum([1 for e in test_hall_values if
                                                   1 - (len(set(e[mitigation])) / len(e[mitigation])) >= threshold])
                    number_of_mitigated_non_hall = sum([1 for e in test_non_hall_values if
                                                        1 - (len(set(e[mitigation])) / len(e[mitigation])) < threshold])
                    if threshold == 0:
                        print(
                            f"{number_of_not_mitigated=} out of {len(test_hall_values)} {[1 - (len(set(e[mitigation])) / len(e[mitigation])) for e in test_hall_values]=},"
                            f"{number_of_mitigated_non_hall=} out of {len(test_non_hall_values)} {[1 - (len(set(e[mitigation])) / len(e[mitigation])) for e in test_non_hall_values]=}")
                elif mitigation == "prob":
                    print(f"{mitigation=} {model_name=} {dataset=} {threshold=}")
                    number_of_not_mitigated = sum([1 for e in test_hall_values if e[mitigation] >= threshold])
                    number_of_mitigated_non_hall = sum([1 for e in test_non_hall_values if e[mitigation] < threshold])
                    print(f"{number_of_not_mitigated=} out of {len(test_hall_values)}")
                else:
                    print(f"{mitigation=} {model_name=} {dataset=} {threshold=}")
                    number_of_not_mitigated = sum([1 for e in test_hall_values if e[mitigation] < threshold])
                    number_of_mitigated_non_hall = sum([1 for e in test_non_hall_values if e[mitigation] >= threshold])
                results[settings][dataset][model_name][mitigation]["Balanced"] = {
                    "hkplus": round(100 * number_of_not_mitigated / len(test_hall_values), 2),
                    "non_hall": round(100 * number_of_mitigated_non_hall / len(test_non_hall_values), 2)}
                assert non_hall_miss == number_of_mitigated_non_hall, f"{non_hall_miss=} {number_of_mitigated_non_hall=}"
                assert hall_miss == number_of_not_mitigated, f"{hall_miss=} {number_of_not_mitigated=}"

        print(results)

        for setting, datasets in results.items():
            for dataset, model_names in datasets.items():
                self.plot_bar_chart(results[setting][dataset], f"{dataset} - {setting}", "Model Name",
                                    "Non-mitigated %",
                                    f"results/{dataset}_{setting}_bar_chart.pdf")

    def plot_bar_chart(self, results, title, x_label, y_label, path):
        # Set a Seaborn theme for better aesthetics
        sns.set_theme(style="darkgrid", font_scale=1.8, rc={
            'font.size': 40,  # Set a large font size
            'axes.titlesize': 40,
            'axes.labelsize': 45,
            'xtick.labelsize': 35,
            'ytick.labelsize': 35,
            'legend.fontsize': 45,
            'figure.figsize': (14, 10),
        })

        fig, ax = plt.subplots()

        # Example input data
        x = np.arange(len(results))
        hkplus = [results[model_name]["temp_generations"]["Balanced"]["hkplus"] for model_name in results.keys()]
        non_hall = [results[model_name]["mean_entropy"]["Balanced"]["hkplus"] for model_name in results.keys()]
        prob = [results[model_name]["prob"]["Balanced"]["hkplus"] for model_name in results.keys()]
        # sns.set_palette("muted")
        # colors = sns.color_palette(["#1f77b4", "#ff7f0e", "#2ca02c"])
        sns.set_palette("colorblind")
        colors = sns.color_palette("colorblind", n_colors=3)
        # Adjust bar width and spacing
        width = 0.25
        ax.bar(x - width, hkplus, width, label='Sampling', color=colors[0], edgecolor="black", hatch="//")
        ax.bar(x, non_hall, width, label='Predictive Entropy', color=colors[1], edgecolor="black", hatch="\\")
        ax.bar(x + width, prob, width, label='Probabilities', color=colors[2], edgecolor="black", hatch="x")

        # Set Y-axis label
        ax.set_ylabel(y_label)

        # Set Y-axis limit
        ax.set_ylim(0, 100)

        # Customize X-axis ticks and labels
        x_labels = [
            model_name.replace("meta-llama_Llama-3.1-8B-Instruct", "Llama-it")
            .replace("google_gemma-2-9b-it", "Gemma-it")
            .replace("mistralai_Mistral-7B-Instruct-v0.3", "Mistral-it")
            .replace("google_gemma-2-9b", "Gemma")
            .replace("meta-llama_Llama-3.1-8B", "Llama")
            .replace("mistralai_Mistral-7B-v0.3", "Mistral")
            for model_name in results.keys()
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, ha="right", rotation=20)  # Rotate for better visibility

        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # Add legend
        ax.legend(loc='best')

        # Adjust layout to prevent overlap
        fig.tight_layout()
        sns.set_context("paper")
        # Save the figure
        plt.savefig(path.replace(".pdf", ".pdf"))
        plt.close()

    def plot_all_measures(self, root_path):
        stats_files = self.find_hallucinations_stats_files(root_path)
        for file_path in stats_files:
            data_hall = self.open_file(file_path)
            print(f"File: {file_path}")
            data_fact = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))
            if data_hall is None or data_fact is None:
                continue
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            settings = file_path.split("/")[3]
            print(f"File: {file_path}")
            for measure in ["prob", "prob_diff", "semantic_entropy", "semantic_entropy_temp_0.5"]:
                threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss = self.get_threshold(
                    data_hall, data_fact, measure)
                assert test_hall_values[0] in data_hall, f"{test_hall_values[0]=} {data_hall[0]=}"
                assert test_non_hall_values[0] in data_fact, f"{test_non_hall_values[0]=} {data_fact[0]=}"
                print(f"{measure=} {threshold=}")
                prob_uncertain = [e[measure] for e in test_hall_values]
                prob_correct = [e[measure] for e in test_non_hall_values]
                self.plot_measure_hallucination_cumulative(prob_correct, prob_uncertain,
                                                           model_name + "_" + dataset + "_" + settings + "_" + measure,
                                                           "results/", measure, threshold)

    def plot_gemmas(self, root_path):
        stats_files = self.find_hallucinations_stats_files(root_path)
        path_gemma_large = ["results/google_gemma-2-27b/triviaqa/alice/hallucinations_stats_half.json",
                            "results/google_gemma-2-27b/triviaqa/child/hallucinations_stats_half.json",
                            "results/google_gemma-2-27b/naturalqa/alice/hallucinations_stats_half.json",
                            "results/google_gemma-2-27b/naturalqa/child/hallucinations_stats_half.json", ]
        for file_path in path_gemma_large:
            data_hall = self.open_file(file_path)
            print(f"File: {file_path}")
            data_fact = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))
            data_hall_small = self.open_file(file_path.replace("27b", "9b"))
            data_fact_small = self.open_file(
                file_path.replace("27b", "9b").replace("_half", "").replace('hallucinations_stats.json',
                                                                            'factuality_stats.json'))
            if data_hall is None or data_fact is None:
                continue
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            settings = file_path.split("/")[3]
            print(f"File: {file_path}")
            max_number_of_examples = len(data_hall)
            # random.seed(42)
            # random.shuffle(data_fact)
            # data_fact = data_fact[:max_number_of_examples]
            for measure in ["prob", "prob_diff", "semantic_entropy"]:
                threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss = self.get_threshold(
                    data_hall, data_fact, measure)
                threshold_small, test_hall_values_small, test_non_hall_values_small, non_hall_miss_small, hall_miss_small = self.get_threshold(
                    data_hall_small, data_fact_small, measure)
                print(f"{measure=} {threshold=}")
                assert test_hall_values[0] in data_hall, f"{test_hall_values[0]=} {data_hall[0]=}"
                assert test_non_hall_values[0] in data_fact, f"{test_non_hall_values[0]=} {data_fact[0]=}"
                prob_hall = [e[measure] for e in test_hall_values]
                prob_correct = [e[measure] for e in test_non_hall_values]
                prob_hall_small = [e[measure] for e in test_hall_values_small]
                prob_correct_small = [e[measure] for e in test_non_hall_values_small]
                self.plot_measure_hallucination_cumulative_gemma(prob_correct=prob_correct, prob_hall=prob_hall,
                                                                 prob_correct_small=prob_correct_small,
                                                                 prob_hall_small=prob_hall_small,
                                                                 title="27vs9_gemma" + "_" + dataset + "_" + settings + "_" + measure,
                                                                 path="results/", measure=measure,
                                                                 threshold_small=threshold_small)

    def plot_measure_hallucination_cumulative(self, prob_uncertain: list, prob_hall: list, title: str, path,
                                              measure: str, true_threshold: float = None):
        """
        This is a graph where the x is the prob and the y is the percentage of examples with higher or equal prob
        :param prob_uncertain: contain the probabilities of the examples
        :param title:
        :return:
        """
        # create 10 bins 1 per 0.1 interval and write the number of examples in each bin
        # Sort probabilities in ascending order
        sns.set_theme(style="darkgrid", font_scale=1.8, rc={
            'font.size': 40,  # Set a large font size
            'axes.titlesize': 40,
            'axes.labelsize': 50,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'legend.fontsize': 50,
            'figure.figsize': (14, 10),
        })
        # Define probability bins from 0 to 1 with a step of 0.05
        bins = np.linspace(1, 0, 21)  # 20 intervals (0 to 1 inclusive with 0.05 steps)
        if "entropy" in measure:
            bins = np.linspace(0, 4, 9)
        print(f"{measure=} {bins=} {true_threshold=} {title=}")
        epsilon = 1e-6
        # Calculate cumulative percentage of examples for each bin
        y_values = [(
                100 * (
                    sum([1 for prob in prob_uncertain if prob >= threshold - epsilon]) / max(1, len(prob_uncertain))))
            for threshold in bins
        ]

        y_values_hall = [
            100 * (sum([1 for prob in prob_hall if prob >= threshold - epsilon]) / max(1, len(prob_hall)))
            for threshold in bins
        ]
        if "entropy" in measure:
            y_values = [
                100 * (sum([1 for prob in prob_uncertain if prob <= threshold + epsilon]) / max(1, len(prob_uncertain)))
                for threshold in bins
            ]

            y_values_hall = [
                100 * (sum([1 for prob in prob_hall if prob <= threshold + epsilon]) / max(1, len(prob_hall)))
                for threshold in bins
            ]
        # sorted_probs = np.sort(prob_uncertain)
        #
        # # Calculate percentage of examples with higher or equal probabilities
        # y_values = 100 * (1 - np.arange(1, len(sorted_probs) + 1) / len(sorted_probs))

        # Plot
        plt.figure(figsize=(14, 10))
        plt.plot(bins, y_values, marker='o', linestyle='-', color='b', markersize=4, label='Correct', linewidth=5)
        plt.plot(bins, y_values_hall, marker='o', linestyle='-', color='r', markersize=4, label='Hallucinate',
                 linewidth=5)
        # plt.title(title)
        plt.grid(True)
        plt.xlim(1, 0)
        if "entropy" in measure:  # from 4 to 0
            plt.xlim(0, 4)
        plt.ylim(0, 100)
        plt.xlabel(measure.replace("_temp_0.5", "").replace("prob_diff", "Probability Difference").replace("prob",
                                                                                                           "Probability").replace(
            "semantic_entropy", "Semantic entropy"))
        plt.ylabel('Cumulative percentage')
        plt.tick_params(axis='both', which='major')
        if "child" in title and "-it" not in title and "Instruct" not in title:
            plt.legend()
        if true_threshold is not None:  # add vertical line for the threshold
            # modify threshold to the closest bin
            threshold = bins[np.argmin(np.abs(bins - true_threshold))]
            plt.axvline(x=threshold, color='k', linestyle='--', label='Threshold')
            # add color to the area between the threshold and the end of the x axis under the hall curve
            if "entropy" in measure:
                plt.fill_between(bins, y_values_hall, 0, where=(bins <= threshold), color='red', alpha=0.3)
            else:
                plt.fill_between(bins, y_values_hall, 0, where=(bins >= threshold), color='red', alpha=0.3)

        # add tags to the x axis every 0.1
        plt.xticks(np.arange(1, -0.001, -0.2))
        if "entropy" in measure:  # from 4 to zero
            plt.xticks(np.arange(0, 4.1, 1))
        plt.yticks(np.arange(0, 101, 20))
        plt.tick_params(axis='x', which='major', pad=20, length=10)  # Add padding for x-axis ticks
        plt.tick_params(axis='y', which='major', pad=20, length=10)  # Add padding for y-axis ticks
        # plt.xticks.size = 50
        # plt.yticks.size = 50
        # plt.tick_params(axis='both', which='major', labelsize=35)
        plt.tight_layout()
        # sns.set_context("paper", font_scale=2)

        # ax = plt.gca()  # Get the current axis
        # ax.spines['top'].set_visible(True)  # Ensure the top spine is visible
        # ax.spines['top'].set_color('black')  # Set the color of the top spine to black
        # ax.spines['top'].set_linewidth(2)  # Set the linewidth of the top spine
        # increase the size of the ticks
        plt.savefig(path + title + ".pdf", format='pdf')
        plt.close()

    def plot_measure_hallucination_cumulative_gemma(self, prob_correct: list, prob_hall: list, title: str, path,
                                                    measure: str, true_threshold: float = None,
                                                    prob_correct_small: list = None, prob_hall_small: list = None,
                                                    threshold_small: float = None):
        """
        This is a graph where the x is the prob and the y is the percentage of examples with higher or equal prob
        :param prob_correct: contain the probabilities of the examples
        :param title:
        :return:
        """
        # create 10 bins 1 per 0.1 interval and write the number of examples in each bin
        # Sort probabilities in ascending order

        # Define probability bins from 0 to 1 with a step of 0.05

        sns.set_theme(style="darkgrid", font_scale=1.8, rc={
            'font.size': 40,  # Set a large font size
            'axes.titlesize': 40,
            'axes.labelsize': 50,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'legend.fontsize': 50,
            'figure.figsize': (14, 10),
        })

        bins = np.linspace(1, 0, 21)  # 20 intervals (0 to 1 inclusive with 0.05 steps)
        if "entropy" in measure:
            bins = np.linspace(0, 4, 9)
        print(f"{measure=} {bins=} {true_threshold=} {title=}")
        epsilon = 1e-6
        # Calculate cumulative percentage of examples for each bin
        y_values = [(
                100 * (sum([1 for prob in prob_correct if prob >= threshold - epsilon]) / max(1, len(prob_correct))))
            for threshold in bins
        ]

        y_values_hall = [
            100 * (sum([1 for prob in prob_hall if prob >= threshold - epsilon]) / max(1, len(prob_hall)))
            for threshold in bins
        ]

        y_values_small = [(
                100 * (sum([1 for prob in prob_correct_small if prob >= threshold_small - epsilon]) / max(1,
                                                                                                          len(prob_correct_small))))
            for threshold_small in bins
        ]

        y_values_hall_small = [
            100 * (sum([1 for prob in prob_hall_small if prob >= threshold_small - epsilon]) / max(1,
                                                                                                   len(prob_hall_small)))
            for threshold_small in bins
        ]

        if "entropy" in measure:
            y_values = [
                100 * (sum([1 for prob in prob_correct if prob <= threshold + epsilon]) / max(1, len(prob_correct)))
                for threshold in bins
            ]

            y_values_hall = [
                100 * (sum([1 for prob in prob_hall if prob <= threshold + epsilon]) / max(1, len(prob_hall)))
                for threshold in bins
            ]

            y_values_small = [
                100 * (sum([1 for prob in prob_correct_small if prob <= threshold_small + epsilon]) / max(1,
                                                                                                          len(prob_correct_small)))
                for threshold_small in bins
            ]

            y_values_hall_small = [
                100 * (sum([1 for prob in prob_hall_small if prob <= threshold_small + epsilon]) / max(1,
                                                                                                       len(prob_hall_small)))
                for threshold_small in bins
            ]
            # sorted_probs = np.sort(prob_uncertain)

        # Plot
        plt.figure(figsize=(14, 10))
        sns.set_palette("colorblind")
        colors = sns.color_palette("colorblind", n_colors=4)
        plt.plot(bins, y_values, marker='o', linestyle='-', color=colors[0], markersize=4, label='Correct-27B',
                 linewidth=5)
        plt.plot(bins, y_values_small, marker='o', linestyle='-', color=colors[1], markersize=4, label='Correct-9B',
                 linewidth=5)
        plt.plot(bins, y_values_hall, marker='o', linestyle='-', color=colors[2], markersize=4, label='Hallucinate-27B',
                 linewidth=5)
        plt.plot(bins, y_values_hall_small, marker='o', linestyle='-', color=colors[3], markersize=4,
                 label='Hallucinate-9B', linewidth=5)
        # plt.title(title)
        plt.grid(True)
        plt.xlim(1, 0)
        if "entropy" in measure:  # from 4 to 0
            plt.xlim(0, 4)
        plt.ylim(0, 100)
        plt.xlabel(measure.replace("_temp_0.5", "").replace("prob_diff", "Probability Difference").replace("prob",
                                                                                                           "Probability").replace(
            "semantic_entropy", "Semantic entropy"))
        plt.ylabel('Cumulative percentage')
        plt.tick_params(axis='both', which='major')
        # if "child" in title and "-it" not in title and "Instruct" not in title:
        if "semantic_entropy" in measure and "child" in title:
            plt.legend(loc='lower right')
        if true_threshold is not None:  # add vertical line for the threshold
            # modify threshold to the closest bin
            threshold = bins[np.argmin(np.abs(bins - true_threshold))]
            plt.axvline(x=threshold, color='k', linestyle='--', label='Threshold')
            # add color to the area between the threshold and the end of the x axis under the hall curve
            if "entropy" in measure:
                plt.fill_between(bins, y_values_hall, 0, where=(bins <= threshold), color='red', alpha=0.3)
            else:
                plt.fill_between(bins, y_values_hall, 0, where=(bins >= threshold), color='red', alpha=0.3)

        # add tags to the x axis every 0.1
        plt.xticks(np.arange(1, -0.001, -0.2))
        if "entropy" in measure:  # from 4 to zero
            plt.xticks(np.arange(0, 4.1, 1))
        plt.yticks(np.arange(0, 101, 20))

        plt.tight_layout()
        # sns.set_context("paper",  font_scale=2)
        plt.tick_params(axis='x', which='major', pad=20, length=15)  # Add padding for x-axis ticks
        plt.tick_params(axis='y', which='major', pad=20, length=15)  # Add padding for y-axis ticks

        plt.savefig(path + title + ".pdf", format='pdf')
        plt.close()

    def threshold_check(self, non_hall_val: list[float], hall_val: list[float], entropy=False):
        """
        Find the threshold between [0,1] that minimizes the number of
        hallucinations (hall) with values higher than the threshold
        and non-hallucinations (non_hall) with values lower than the threshold.

        Returns:
            threshold (float): The optimal threshold.
            num_misclassified_hall (int): Number of hall misclassified as non_hall.
            num_misclassified_non_hall (int): Number of non_hall misclassified as hall.
        """
        all_values = sorted(set(non_hall_val + hall_val))
        assert len(all_values) <= len(non_hall_val) + len(
            hall_val), f"{len(all_values)=} {len(non_hall_val)=} {len(hall_val)=}"
        for val in all_values:
            assert val in non_hall_val or val in hall_val, f"{val=}"
        # print(f"len all_values: {all_values}")
        best_threshold = 0.0
        min_misclassifications = float('inf')
        num_misclassified_hall = 0
        num_misclassified_non_hall = 0

        for threshold in all_values:
            # Count misclassifications for the current threshold
            misclassified_hall = sum(h > threshold for h in hall_val)
            misclassified_non_hall = sum(n <= threshold for n in non_hall_val)
            if entropy:
                misclassified_hall = sum(h < threshold for h in hall_val)
                misclassified_non_hall = sum(n >= threshold for n in non_hall_val)
            total_misclassified = misclassified_hall + misclassified_non_hall

            # Update the optimal threshold if fewer misclassifications are found
            if total_misclassified < min_misclassifications:
                min_misclassifications = total_misclassified
                best_threshold = threshold
                num_misclassified_hall = misclassified_hall
                num_misclassified_non_hall = misclassified_non_hall
        return best_threshold, num_misclassified_hall, num_misclassified_non_hall

    def get_threshold(self, data_hall, data_non_hall, parameter):
        """
        Find the threshold between [0,1] that minimizes the number of
        hallucinations (hall) with values higher than the threshold
        and non-hallucinations (non_hall) with values lower than the threshold.

        Returns:
            threshold (float): The optimal threshold.
            test_hall_values (list): Hallucinations that were not used for validation.
            test_non_hall_values (list): Non-hallucinations that were not used for validation."""
        hall_values = []
        non_hall_values = []
        size = int(min(len(data_hall), len(data_non_hall)))
        # random.seed(42)
        # random.shuffle(data_hall)
        # random.shuffle(data_non_hall)
        hall_values = data_hall[:size]
        non_hall_values = data_non_hall[:size]
        test_hall_values = data_hall[:size]
        test_non_hall_values = data_non_hall[:size]
        assert len(hall_values) == len(non_hall_values), f"{len(hall_values)=} {len(non_hall_values)=}"
        print(f" {len(test_hall_values)=} {len(test_non_hall_values)=}")

        hall_values = [e[parameter] for e in hall_values]
        non_hall_values = [e[parameter] for e in non_hall_values]
        if "temp_generations" in parameter:
            hall_values = [1 - (len(set(e)) / len(e)) for e in hall_values]
            non_hall_values = [1 - (len(set(e)) / len(e)) for e in non_hall_values]
        # print(f"len hall_values: {len(hall_values)} {len(non_hall_values)=}")
        # print(f"{hall_values[:5]=} {non_hall_values[:5]=}")
        threshold, _, _ = self.threshold_check(non_hall_values, hall_values,
                                               entropy=True if "entropy" in parameter else False)
        # check accuracy on test set
        if "temp_generations" in parameter:
            non_hall_miss = sum(
                [1 for e in test_non_hall_values if 1 - (len(set(e[parameter])) / len(e[parameter])) < threshold])
            hall_miss = sum(
                [1 for e in test_hall_values if 1 - (len(set(e[parameter])) / len(e[parameter])) >= threshold])
        elif "entropy" in parameter:
            non_hall_miss = sum([1 for e in test_non_hall_values if e[parameter] >= threshold])
            hall_miss = sum([1 for e in test_hall_values if e[parameter] < threshold])
        else:
            non_hall_miss = sum([1 for e in test_non_hall_values if e[parameter] < threshold])
            hall_miss = sum([1 for e in test_hall_values if e[parameter] >= threshold])
            print(f"{non_hall_miss=} {hall_miss=}")
        return threshold, test_hall_values, test_non_hall_values, non_hall_miss, hall_miss

    def similarity_between_settings(self, root_path, shared_prompts=False):
        print(f"shared_prompts: {shared_prompts}")
        stats_files = self.find_hallucinations_stats_files(root_path)
        all_data = {}
        random.seed(42)
        for file_path in stats_files:
            if "alice" in file_path or "child2" in file_path or "27" in file_path:
                continue
            data_child = self.open_file(file_path)
            data_fact_child = self.open_file(
                file_path.replace("_half", "").replace('hallucinations_stats.json', 'factuality_stats.json'))
            data_alice = self.open_file(file_path.replace("child", "alice"))
            data_fact_alice = self.open_file(
                file_path.replace("child", "alice").replace("_half", "").replace('hallucinations_stats.json',
                                                                                 'factuality_stats.json'))
            if data_child is None or data_alice is None:
                continue
            parameter = "prob"
            # parameter = "semantic_entropy"
            threshold_child, test_hall_values_child, test_non_hall_values_child, non_hall_miss_child, hall_miss_child = self.get_threshold(
                data_child, data_fact_child, parameter)
            threshold_alice, test_hall_values_alice, test_non_hall_values_alice, non_hall_miss_alice, hall_miss_alice = self.get_threshold(
                data_alice, data_fact_alice, parameter)
            # calculate spearman correlation between the two hallucinations

            if "entropy" in parameter:
                certain_child_examples = [e for e in test_hall_values_child if e[parameter] <= threshold_child]
                certain_alice_examples = [e for e in test_hall_values_alice if e[parameter] <= threshold_alice]
            else:
                certain_child_examples = [e for e in test_hall_values_child if e[parameter] >= threshold_child]
                certain_alice_examples = [e for e in test_hall_values_alice if e[parameter] >= threshold_alice]
            low_certain_child = sorted([e for e in test_hall_values_child if e[parameter] < threshold_child],
                                       key=lambda x: x[parameter])[:len(certain_child_examples)]
            low_certain_alice = sorted([e for e in test_hall_values_alice if e[parameter] < threshold_alice],
                                       key=lambda x: x[parameter])[:len(certain_alice_examples)]
            assert low_certain_child[0][parameter] <= low_certain_child[1][parameter]
            assert len(low_certain_child) <= len(certain_child_examples)
            assert len(low_certain_alice) <= len(certain_alice_examples)
            print(
                f"{len(certain_child_examples)=} {len(certain_alice_examples)=} {len(low_certain_child)=} {len(low_certain_alice)=}")
            set_low_certain_child = set([e["prompt"].split("question:")[-1] for e in low_certain_child])
            set_low_certain_alice = set([e["prompt"].split("question:")[-1] for e in low_certain_alice])
            set_prompt_child = set([e["prompt"].split("question:")[-1] for e in test_hall_values_child])
            assert len(set_prompt_child) == len(test_hall_values_child)
            set_prompt_alice = set([e["prompt"].split("question:")[-1] for e in test_hall_values_alice])
            assert len(set_prompt_alice) == len(test_hall_values_alice)
            shared = set_prompt_child.intersection(set_prompt_alice)
            assert len(shared) <= len(set_prompt_child) and len(shared) <= len(set_prompt_alice)
            shared_low_certain = set([e["prompt"].split("question:")[-1] for e in low_certain_child]).intersection(
                set([e["prompt"].split("question:")[-1] for e in low_certain_alice]))
            y_options = np.array(
                [e[parameter] for e in test_hall_values_child if e["prompt"].split("question:")[-1] in shared])
            y_all_values = [e["prompt"].split("question:")[-1] for e in test_hall_values_alice]
            x_options = np.array(
                [test_hall_values_alice[y_all_values.index(e["prompt"].split("question:")[-1])][parameter] for i, e in
                 enumerate(test_hall_values_child) if e["prompt"].split("question:")[-1] in shared])

            correlation = spearmanr(x_options, y_options).correlation
            print(f"{file_path=} {correlation=}")

            # print(f"who question {sum([1 for e in test_hall_values_child if 'who' in e['prompt']])/len(test_hall_values_child)} {sum([1 for e in test_hall_values_alice if 'who' in e['prompt']])/len(test_hall_values_alice)} {sum([1 for e in certain_child_examples if 'who' in e['prompt']])/len(certain_child_examples)} {sum([1 for e in certain_alice_examples if 'who' in e['prompt']])/len(certain_alice_examples)}")

            set_prompt_alice_certain = set([e["prompt"].split("question:")[-1] for e in certain_alice_examples])
            set_prompt_child_certain = set([e["prompt"].split("question:")[-1] for e in certain_child_examples])
            set_prompt_child_certain_shared = set_prompt_child_certain.intersection(shared)
            set_prompt_alice_certain_shared = set_prompt_alice_certain.intersection(shared)
            sim_certain = set_prompt_alice_certain.intersection(set_prompt_child_certain)
            sim_examples = set_prompt_alice.intersection(set_prompt_child)

            print(f"{file_path=} {round(len(sim_examples) / len(set_prompt_alice.union(set_prompt_child)), 2)}")

            # Permutation test
            observed_jaccard = len(sim_certain) / len(set_prompt_alice_certain.union(set_prompt_child_certain))
            observed_jaccard_shared = len(
                set_prompt_child_certain_shared.intersection(set_prompt_alice_certain_shared)) / len(
                set_prompt_child_certain_shared.union(set_prompt_alice_certain_shared))
            num_permutations = 10000
            random.seed(42)
            flag_compar_low_certainty = False
            p_value_both = []
            mean_permuted_jaccard = []
            for k in range(2):
                shared_ = k == 1
                permuted_jaccards = []
                if shared_:
                    size = max(len(set_prompt_alice_certain_shared), len(set_prompt_child_certain_shared))
                else:
                    size = max(len(set_prompt_alice_certain), len(set_prompt_child_certain))
                assert shared_ and size < len(shared) or not shared_ and size < len(set_prompt_child)
                for _ in range(num_permutations):
                    if shared_:
                        perm_random_child = set(e for e in random.sample(shared, len(set_prompt_child_certain_shared)))
                        perm_random_alice = set(e for e in random.sample(shared, len(set_prompt_alice_certain_shared)))
                        assert len(perm_random_child) == len(set_prompt_child_certain_shared)
                        assert len(perm_random_alice) == len(set_prompt_alice_certain_shared)

                    else:
                        perm_random_child = set(
                            e for e in random.sample(set_prompt_child, len(set_prompt_child_certain)))
                        perm_random_alice = set(
                            e for e in random.sample(set_prompt_alice, len(set_prompt_alice_certain)))
                        assert len(perm_random_child) == len(set_prompt_child_certain)
                        assert len(perm_random_alice) == len(set_prompt_alice_certain)
                        if flag_compar_low_certainty:
                            perm_random_child = set(
                                e for e in random.sample(set_low_certain_child, len(set_prompt_child_certain)))
                            perm_random_alice = set(
                                e for e in random.sample(set_low_certain_alice, len(set_prompt_alice_certain)))
                            assert len(perm_random_child) == len(set_prompt_child_certain)
                            assert len(perm_random_alice) == len(set_prompt_alice_certain)

                    permuted_sim_certain = perm_random_child.intersection(perm_random_alice)
                    permuted_jaccard = len(permuted_sim_certain) / len(perm_random_child.union(perm_random_alice))
                    permuted_jaccards.append(permuted_jaccard)
                assert len(permuted_jaccards) == num_permutations

                if shared_:

                    p_value = sum(1 for j in permuted_jaccards if j >= observed_jaccard_shared) / num_permutations
                else:
                    p_value = sum(1 for j in permuted_jaccards if j >= observed_jaccard) / num_permutations
                p_value_both.append(p_value)
                mean_permuted_jaccard.append((np.mean(permuted_jaccards), np.std(permuted_jaccards)))

            print(
                f"{file_path=} {size} {max(len(set_prompt_alice_certain), len(set_prompt_child_certain))} Observed Jaccard: {round(observed_jaccard, 2)}, under shared {round(observed_jaccard_shared, 2)}, P-value: {p_value_both}, mean{np.mean(permuted_jaccards)}")
            model_name = file_path.split("/")[1]
            dataset = file_path.split("/")[2]
            if all_data.get(model_name) is None:
                all_data[model_name] = {}
            all_data[model_name][dataset] = {"jaccard": round(observed_jaccard * 100, 2),
                                             "jaccard_shared": round(observed_jaccard_shared * 100, 2),
                                             "p_value": round(p_value_both[0], 4),
                                             "p_value_shared": round(p_value_both[1], 4),
                                             "mean_permuted": mean_permuted_jaccard[0],
                                             "mean_permuted_shared": mean_permuted_jaccard[1]}
        print(all_data)
        # table of the results
        latex_table = """\\begin{table}[h!]
        \\centering
        \\begin{tabular}{|l|c|c|c|}
        \\hline
        Model&Dataset&Random& Certainty\\\\
        \\hline"""
        for model_name, datasets in all_data.items():
            current_model_name = model_name.replace("google_gemma-2-9b", "Gemma").replace("google_gemma-2-9b-it",
                                                                                          "Gemma-Inst").replace(
                "meta-llama_Llama-3.1-8B", "Llama").replace("mistralai_Mistral-7B-Instruct-v0.3",
                                                            "Mistral-Inst").replace("mistralai_Mistral-7B-v0.3",
                                                                                    "Mistral")
            latex_table += (
                        "\multirow{2}{*}{" + f"{current_model_name}" + "}" + f" & TriviaQA & {round(datasets['triviaqa']['mean_permuted'][0] * 100, 2)} &" + "\\textbf{" + f"{datasets['triviaqa']['jaccard']}" + "}" + f"\\\\ &NQ&{round(datasets['naturalqa']['mean_permuted'][0] * 100, 2)} & " + "\\textbf{" + f"{datasets['naturalqa']['jaccard']}" + "}" + f"\\\\\\midrule")
        latex_table += """\n\\end{tabular}
        \\caption{Results for Jaccard similarity between settings across TriviaQA and NaturalQA datasets.}
        \\label{tab:results}
        \\end{table}"""
        print(latex_table)

        latex_table = """\\begin{table}[h!]
                \\centering
                \\begin{tabular}{|l|c|c|c|}
                \\hline
                Model &Dataset&Random& Certainty\\\\
                \\hline"""
        for model_name, datasets in all_data.items():
            current_model_name = model_name.replace("google_gemma-2-9b", "Gemma").replace("google_gemma-2-9b-it",
                                                                                          "Gemma-Inst").replace(
                "meta-llama_Llama-3.1-8B", "Llama").replace("mistralai_Mistral-7B-Instruct-v0.3",
                                                            "Mistral-Inst").replace("mistralai_Mistral-7B-v0.3",
                                                                                    "Mistral")
            latex_table += (
                        "\multirow{2}{*}{" + f"{current_model_name}" + "}" + f"& TriviaQA  & {round(datasets['triviaqa']['mean_permuted_shared'][0] * 100, 2)} &" + "\\textbf{" + f"{datasets['triviaqa']['jaccard_shared']}" + "}" + f"\\\\ &NQ&{round(datasets['naturalqa']['mean_permuted_shared'][0] * 100, 2)} & " + "\\textbf{" + f"{datasets['naturalqa']['jaccard_shared']}" + "}" + f"\\\\\\midrule")
        latex_table += """\n\\end{tabular}
                \\caption{Results for Jaccard similarity between settings across TriviaQA and NaturalQA datasets.}
                \\label{tab:results}
                \\end{table}"""
        print(latex_table)


def run_results():
    root_path = "results/"
    cs = case_study(root_path)
    cs.remove_half_hallucinations(root_path)
    cs.plot_all_measures(root_path)
    cs.plot_gemmas(root_path)
    cs.mitigation_uncertainty_based(root_path)
    cs.similarity_between_settings(root_path, shared_prompts=False)
