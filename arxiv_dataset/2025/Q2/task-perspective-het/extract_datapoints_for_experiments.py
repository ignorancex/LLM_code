import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import argparse




def parse_args():

    argparser = argparse.ArgumentParser("Extracting scene classes from client datasets")
    
    argparser.add_argument('--client_file_name', type=str, help="Path to the client information .csv file")
    argparser.add_argument('--data_path', type=str, help="Path to the data files")
    argparser.add_argument('--taskonomy_clients_to_include', type=int, help="Number of taskonomy clients to include")
    argparser.add_argument('--scene_class_file_name', type=str, help="Name of the scene class file")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    
    args = parse_args()
    cfp = os.path.join(args.data_path, args.client_file_name)
    client_df = pd.read_csv(cfp)
    client_list = client_df.id.values[:args.taskonomy_clients_to_include]

    total_classes_df = pd.DataFrame(data=[], columns=["filepath", "class_idx"])

    for client in client_list:
        # Read client csv file
        client_list_fp = os.path.join(args.data_path, f"class_scene/taskonomy/", client, "scene_classes.csv")
        client_classes = pd.read_csv(client_list_fp, delimiter=",")
        # print(client_classes.filepath.values[:5])
        # raise KeyboardInterrupt("Stop here")
        # Modify the filepath column
        client_classes['filepath'] = client_classes['filepath'].map(lambda x: f"{args.data_path}/rgb/taskonomy/{client}/{x}")
        # Append the modified df to the total df
        total_classes_df = pd.concat((total_classes_df, client_classes), axis=0, ignore_index=True)
    
    # print(total_classes_df.filepath.values[:5])
    
    top_classes = total_classes_df.class_idx.value_counts().index[:16]
    top_classes_df = total_classes_df[total_classes_df.class_idx.isin(top_classes)]

    class_labels = np.genfromtxt(os.path.join(args.data_path, args.scene_class_file_name), dtype='U18')
    top_class_labels = class_labels[top_classes]
    print(f"top class labels:\n{top_class_labels}")

    plt.figure()
    top_classes_df.class_idx.value_counts().plot(kind='bar', colormap='viridis')
    plt.grid()
    plt.xticks(range(len(top_class_labels)), top_class_labels, rotation=90)
    
    plt.title("Number of datapoints per class")
    plt.tight_layout()
    plt.savefig("./paper_figs/class_data_dist.png")

    # save the filepath for the datapoints with top classes to csv file
    top_classes_df.to_csv(os.path.join(args.data_path, "filtered_file_paths.csv"), index=False)