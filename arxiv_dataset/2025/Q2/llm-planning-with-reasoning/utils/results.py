import re
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm


datalabels = ['Easy' , 'Medium', 'Hard', 'ExtraHard', 'All']
exact_metrics = ['execAcc', 'execMatch']
exact_titles = ["Execution Accuracy", "Exact Matching Accuracy"]
partial_metrics = ['partialAcc', 'partialRecall', 'partialF1']
partial_titles = ["Partial Matching Accuracy", "Partial Matching Recall", "Partial Matching F1"]
matching_title = "Matching Results"
matching_labels = ["Exact Acc","Partial Acc", "Partial Recall", "Partial F1"]
exect_matching_metric =  'execMatch'

# parse data from text
def parse_text(text):
    def extract_section(pattern, text):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            rows = re.findall(r"([\w/() ]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", match.group(1))
            return {row[0].strip(): [float(num) for num in row[1:]] for row in rows}
        return {}

    patterns = {
        "execAcc": r"execution\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        "execMatch": r"exact match\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        "partialAcc": r"PARTIAL MATCHING ACCURACY(.*?)PARTIAL MATCHING RECALL",
        "partialRecall": r"PARTIAL MATCHING RECALL(.*?)PARTIAL MATCHING F1",
        "partialF1": r"PARTIAL MATCHING F1(.*?)$"
    }

    results = {}

    # Extract single-line values (execAcc, execMatch)
    for key in exact_metrics:
        match = re.search(patterns[key], text)
        if match:
            results[key] = [float(match.group(i)) for i in range(1, 6)]
        else:
            results[key] = None

    # Extract multi-row values (partialAcc, partialRecall, partialF1)
    for key in partial_metrics:
        results[key] = extract_section(patterns[key], text)

    return results

# display parsed data
def disp_parsed_data(parsed_data):
    # Display results
    print("Execution Accuracy:", parsed_data["execAcc"])
    print("Exact Matching Accuracy:", parsed_data["execMatch"])
    print("\nPartial Matching Accuracy:")
    for key, values in parsed_data["partialAcc"].items():
        print(f"  {key}: {values}")

    print("\nPartial Matching Recall:")
    for key, values in parsed_data["partialRecall"].items():
        print(f"  {key}: {values}")

    print("\nPartial Matching F1:")
    for key, values in parsed_data["partialF1"].items():
        print(f"  {key}: {values}")

# given text file names, get the parsed data
def parse_results(result_fnames):
    results = {}
    for key in result_fnames.keys():
        results_fname = result_fnames[key]
        # open text file
        with open(results_fname, "r", encoding="utf-8") as file:
            text = file.read()
        # parse
        parsed_data = parse_text(text)
        # collect
        results[key] = parsed_data
    return results

# Plots a bar chart comparing multiple vectors. vectors[group][metric]
def plot_vector_comparison(vectors, labels=None, title="Comparison of Vectors", groups=None):
    """
    Plots a bar chart comparing multiple vectors. converts into % by multiplying with 100.
    
    Parameters:
    - vectors: list of lists/arrays where each inner list represents a vector of numerical values
    - labels: list of labels for each index (default: Item 1, Item 2, ...)
    - title: title of the plot
    - groups: list of group names corresponding to each vector
    """
    
    # scale the vectors
    vectors = deepcopy(vectors)
    for i in range(len(vectors)):
        for j in range(len(vectors[i])):
            vectors[i][j] *= 100
    
    num_vectors = len(vectors)
    vector_length = len(vectors[0])
    
    # colormap
    cmap = cm.get_cmap('cividis', num_vectors)  # Choose 'plasma' or 'cividis' if preferred

    # Ensure all vectors have the same length
    if not all(len(v) == vector_length for v in vectors):
        raise ValueError("All vectors must have the same length.")
    
    x = np.arange(vector_length)  # X locations for the bars
    width = 0.8 / num_vectors  # Dynamically set width to fit all bars

    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Assign default group names if not provided
    if groups is None:
        groups = [f'G{i+1}' for i in range(num_vectors)]

    minv = 1
    # Plot each vector
    for i, (vector, group) in enumerate(zip(vectors, groups)):
        ax.bar(x + (i - num_vectors/2) * width, vector, width, label=group, color=cmap(i))
        minv = min(minv, min(vector))

    ax.set_xlabel('Problem difficulty')
    ax.set_ylabel(title+" (%)")
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_ylim([minv*0.8,100])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    if labels is None:
        labels = [f'Item {i+1}' for i in x]
    
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    
    plt.show()

    print(f"\n========{title}=============\n\n")
    print_vector_comparison_table(vectors, labels=labels, groups=groups)


def plot_vector_comparisonV2(vectors, labels=None, title="Comparison of Vectors", groups=None):
    """
    Plots a row of bar charts comparing multiple vectors for each label separately.
    
    Parameters:
    - vectors: list of lists/arrays where each inner list represents a vector of numerical values
    - labels: list of labels for each index (default: Item 1, Item 2, ...)
    - title: title of the plot
    - groups: list of group names corresponding to each vector
    """
    num_vectors = len(vectors)
    vector_length = len(vectors[0])

    # Ensure all vectors have the same length
    if not all(len(v) == vector_length for v in vectors):
        raise ValueError("All vectors must have the same length.")

    # Create a deep copy to avoid modifying the original vectors
    vectors = deepcopy(vectors)
    for i in range(len(vectors)):
        for j in range(len(vectors[i])):
            vectors[i][j] *= 100

    # Assign default group names if not provided
    if groups is None:
        groups = [f'G{i+1}' for i in range(num_vectors)]

    # Assign default labels if not provided
    if labels is None:
        labels = [f'Item {i+1}' for i in range(vector_length)]

    # Set up colormap for distinct colors
    cmap = cm.get_cmap('viridis', num_vectors)
    colors = [cmap(i) for i in range(num_vectors)]

    # Create subplots (one per label)
    fig, axes = plt.subplots(1, vector_length, figsize=(vector_length * 3, 4), sharey=False)
    fig.suptitle(title)

    for idx, ax in enumerate(axes):
        x = np.arange(num_vectors)  # X locations for bars
        bar_width = 0.8  

        # Get values for this specific label across all groups
        values = [vector[idx] for vector in vectors]

        # Plot bars for this label
        ax.bar(x, values, width=bar_width, color=colors)

        ax.set_title(labels[idx])
        ax.set_xticks(x)
        # ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylim([min(values)*0.95,max(values)*1.05])

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Create a single legend for all subplots
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_vectors)]
    fig.legend(handles, groups, loc='upper right', frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title and legend
    plt.show()

def print_vector_comparison_table(vectors, labels=None, groups=None):
    """
    Prints a Markdown table where labels are columns and groups are rows.
    
    Parameters:
    - vectors: list of lists/arrays where each inner list represents a vector of numerical values
    - labels: list of labels for each index (default: Item 1, Item 2, ...)
    - groups: list of group names corresponding to each vector
    """
    num_vectors = len(vectors)
    vector_length = len(vectors[0])

    # Ensure all vectors have the same length
    if not all(len(v) == vector_length for v in vectors):
        raise ValueError("All vectors must have the same length.")

    # Assign default labels if not provided
    if labels is None:
        labels = [f'Item {i+1}' for i in range(vector_length)]

    # Assign default group names if not provided
    if groups is None:
        groups = [f'G{i+1}' for i in range(num_vectors)]

    # Print header
    header = "| Group | " + " | ".join(labels) + " |"
    separator = "|-" + "-|".join(["-" * len(label) for label in labels]) + "-|"

    print(header)
    print(separator)

    # Print each row
    for i, group in enumerate(groups):
        row = f"| {group} | " + " | ".join(f"{vectors[i][j]:.2f}" for j in range(vector_length)) + " |"
        print(row)


def plot_accuracy(results, groups):
    for i in range(len(exact_metrics)):
        metric = exact_metrics[i]
        plotdata = [results[r][metric] for r in results.keys()]
        plot_vector_comparison(plotdata,title=exact_titles[i],labels=datalabels ,groups=groups)

def plot_partial_matching(results, groups):
    for i in range(len(partial_metrics)): # each metric
        metric = partial_metrics[i]
        plotdata = []
        for g in results.keys(): # each group
            dmap = results[g][metric]
            dvals = [dmap[key] for key in dmap.keys()]
            avgVals = np.mean(np.array(dvals), axis=0)
            plotdata.append(avgVals)
        plot_vector_comparison(plotdata,title=partial_titles[i],labels=datalabels ,groups=groups)

def plot_matching(results, groups,subClass_indx):
    
    # plot all metrics but for all sub-classes
    plotdata = []
    for g in results.keys(): # each group
        grp_plotdata = []
        
        #exact result
        exact_acc = results[g][exect_matching_metric][subClass_indx] # for all sub-class
        grp_plotdata.append(exact_acc)

        #partial result
        for i in range(len(partial_metrics)): # each metric
            metric = partial_metrics[i]
            dmap = results[g][metric]
            dvals = [dmap[key] for key in dmap.keys()]
            avgVals = np.mean(np.array(dvals), axis=0)
            grp_plotdata.append(avgVals[subClass_indx]) # get the last one which is for all sub-class
        plotdata.append(grp_plotdata)
    
    plot_vector_comparison(plotdata,title=matching_title ,labels=matching_labels ,groups=groups)