# #!/bin/sh

# Agg_throughput=0;

# throughput=`cat ../k-bench-results/**/dp_redis_density/dp_redis_density-*/memtier.out  | grep "Totals" | awk {'print $2'}`
# echo "Throughput of pod $num is $throughput";
# Agg_throughput=`echo "$throughput + $Agg_throughput" | bc`
# echo "Aggregate throughput = $Agg_throughput";



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

distributions = ['k0s', 'k3s', 'k8s', 'kubeEdge', 'openYurt']
values = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

# Create the dictionary
mapping = dict(zip(distributions, values))


def get_metrics(file_path):
    ops_sec = []
    latency = []
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Check if the line contains 'Totals'
            if 'Totals' in line:
                # Split the line into columns
                columns = line.split()
                # Check if there are at least two columns
                if len(columns) >= 2:
                    # Append the second column value to the list
                    ops_sec.append(float(columns[1]))
                    latency.append(float(columns[4]))
    return ops_sec, latency

def create_plots(files, title, xlabel, toSave=False):
    all_data = []

    for file in files:
        path_components = file.split('/')
        dist = path_components[2]
        # data is DataFrame
        data = pd.DataFrame()
        data['ops_sec'], data['latency'] = get_metrics(file)
        data['dist'] = mapping[dist]
        all_data.append(data)
    # print(all_data)
    combined_data = pd.concat(all_data, ignore_index=True)

    # plt.figure(figsize=(6, 6))
    fig, ax1 = plt.subplots(figsize=(6,5))
    # whiskerprops1 = dict(color='red', linewidth=10)
    boxprops1 = dict(linewidth=3, fill=None, color='red')
    capprops1 = dict(linewidth=3, color='red')
    medianprops1 = dict(linestyle='-', linewidth=3, color='red')
    sns.boxplot(data=combined_data, x='dist', y='latency', ax=ax1, color='red', boxprops=boxprops1, capprops=capprops1, medianprops=medianprops1 )#, width=0.5)
    ax1.set_ylabel('Avg Latency (ms)', color='red', fontsize='x-large')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel(xlabel)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    ax2 = ax1.twinx()
    # whiskerprops2 = dict(color='blue', linewidth=10)
    boxprops2 = dict(linewidth=3, fill=None, color='blue')
    capprops2 = dict(linewidth=3, color='blue')
    medianprops2 = dict(linestyle='-', linewidth=3, color='blue')
    sns.boxplot(data=combined_data, x='dist', y='ops_sec', ax=ax2, color='blue', boxprops=boxprops2, capprops=capprops2, medianprops=medianprops2 )#, width=0.5)
    ax2.set_ylabel('Throughput operations (Ops/sec)', color='blue', fontsize='x-large')
    ax2.tick_params(axis='y', labelcolor='blue')
    fig.tight_layout()  
    plt.grid(True)
    # plt.title(f'{title}')

    if toSave:
        plt.savefig(f'../diagrams/data-plane-latency-throughput.pdf', format='pdf')
    else:
        plt.show()



toSave = True

files = []
for dist in distributions:
    for i in range(1, 5):
        files.append(f'../k-bench-results/{dist}/dp_redis_density/dp_redis_density-{i}/memtier.out')
create_plots(files, 'Data plane - Latency/Throughput', 'Distributions', toSave)