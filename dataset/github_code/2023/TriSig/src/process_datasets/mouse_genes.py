import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import DI2 as di2
import numpy as np
import os.path as path

curr_dir = os.getcwd()
data_folder = path.abspath(path.join(__file__, "../../.."))+"\\data\\mouse_genes\\"

######## Mouse data ########
original_mouse_files = [
    data_folder+"TCDD_study_Percellome_dataPart1of4.csv",
    data_folder+"TCDD_study_Percellome_dataPart2of4.csv",
    data_folder+"TCDD_study_Percellome_dataPart3of4.csv",
    data_folder+"TCDD_study_Percellome_dataPart4of4.csv"
]


def read_mouse_data(files, transpose=True):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i])
        df.columns = [i for i in range(len(df.columns))]
        dataframes.append(df.drop(columns=[df.columns[0]]).T if transpose else df.drop(columns=[df.columns[0]]))
    return dataframes


def process_mouse_data(files):
    dataframes = []
    for f_i in range(len(files)):
        df = pd.read_csv(files[f_i])
        df = df.drop(columns=["Descriptions"])
        for col in df.columns:
            if "_Detection" in col:
                df = df.drop(columns=[col])
        cols = list(df[df.columns[0]])
        df = df.drop(columns=[df.columns[0]])
        df = df.T
        df.columns = cols
        dataframes.append(df)

    combined_df = dataframes[0]
    for df_i in range(1,len(dataframes)):
        combined_df = pd.concat([combined_df, dataframes[df_i]], axis=0)
    column_variance = []
    for col in combined_df.columns:
        column_variance.append([col, np.var(combined_df[col])])
    column_variance = sorted(column_variance, key=lambda val: val[1], reverse=True)
    for df_i in range(len(dataframes)):
        dataframes[df_i] = dataframes[df_i].drop(columns=[val[0] for val in column_variance[500:]])
    combined_df = pd.concat(dataframes).reset_index(drop=True)

    # min max norm
    n_array = combined_df.to_numpy()
    maximum = np.amax(n_array)
    minimum = np.amin(n_array)
    real_dataframes = np.array_split(combined_df, 4)
    for row_i in range(len(real_dataframes[0][real_dataframes[0].columns[0]])):
        for col in real_dataframes[0].columns:
            min_loc = max(minimum, real_dataframes[0].iloc[row_i][col]/2)
            max_loc = max(maximum, real_dataframes[3].iloc[row_i][col]*2)
            for df_i in range(len(real_dataframes)):
                real_dataframes[df_i].iloc[row_i][col] = (real_dataframes[df_i].iloc[row_i][col] - min_loc) / (max_loc - min_loc)

    for df_i in range(len(dataframes)):
        dataframes[df_i].to_csv(curr_dir+"\\mouse_genes_output\\real_"+str(df_i)+".csv")

    combined_df = pd.concat(real_dataframes).reset_index(drop=True)
    combined_df = di2.distribution_discretizer(combined_df, number_of_bins=3, cutoff_margin=0.0, single_column_discretization=True)
    discretized_dataframes = np.array_split(combined_df, 4)

    for df_i in range(len(discretized_dataframes)):
        discretized_dataframes[df_i].to_csv(curr_dir+"\\mouse_genes_output\\discretized_"+str(df_i)+".csv")


def transform_mouse_zaki(files):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i])
        df.columns = [i for i in range(len(df.columns))]
        dataframes.append(df.drop(columns=[df.columns[0]]).T)

    output = open(curr_dir+"\\mouse_genes_output\\mouse_transformed_diogo.tab", "w")
    output.write("Total Time:\t"+str(len(dataframes))+"\n")
    output.write("Total Samples:\t"+str(len(dataframes[0].columns))+"\n")
    output.write("Total Genes:\t"+str(len(dataframes[0][dataframes[0].columns[0]]))+"\n")

    for z_i in range(len(dataframes)):
        output.write("Time\t"+str(z_i)+"\n")
        output.write("ID\tNAME")
        for col_i in range(len(dataframes[z_i].columns)):
            output.write("\tS-"+str(col_i))
        output.write("\n")
        for x_i in range(len(dataframes[z_i][dataframes[z_i].columns[0]])):
            output.write(str(x_i)+"\tG-"+str(x_i))
            for y_i in range(len(dataframes[z_i].columns)):
                output.write("\t"+str(dataframes[z_i].iloc[x_i][dataframes[z_i].columns[y_i]]))
            output.write("\t\n")

    output.close()

real_mouse_files = [
    curr_dir+"\\process_datasets\\mouse_genes_output\\real_0.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\real_1.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\real_2.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\real_3.csv"
]

discretized_mouse_files = [
    curr_dir+"\\process_datasets\\mouse_genes_output\\discretized_0.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\discretized_1.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\discretized_2.csv",
    curr_dir+"\\process_datasets\\mouse_genes_output\\discretized_3.csv"
]

if __name__ == "__main__":
    process_mouse_data(original_mouse_files)



    transform_mouse_zaki(discretized_mouse_files)


