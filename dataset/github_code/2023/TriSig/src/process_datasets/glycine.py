import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import DI2 as di2
import os.path as path

curr_dir = os.getcwd()
data_folder = path.abspath(path.join(__file__, "../../.."))+"\\data\\glycine\\"

######## Glycine data ########

glycine_file = data_folder+"data.csv"


def read_glycine(files, transpose=True):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i])
        dataframes.append(df.drop(columns=[df.columns[0], df.columns[-1]]).T if transpose else df.drop(columns=[df.columns[0], df.columns[-1]]))
    return dataframes


def process_glycine(file):
    df = pd.read_csv(file)

    for col in df.columns:
        if col not in ["Sample Name", "Case", "Treatment", "Time"]:
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

    df_discretized = df.copy()
    for col in df_discretized.columns:
        if col not in ["Sample Name", "Case", "Treatment", "Time"]:
            df_discretized[col] = di2.distribution_discretizer(df_discretized[col].to_frame(), number_of_bins=5, cutoff_margin=0.0, single_column_discretization=True).iloc[:, 0]

    dataframes = [
        df[df["Time"]==-1],
        df[df["Time"]==0],
        df[df["Time"]==1],
        df[df["Time"]==2],
        df[df["Time"]==3],
        df[df["Time"]==4]
    ]

    dataframes_disc = [
        df_discretized[df_discretized["Time"]==-1],
        df_discretized[df_discretized["Time"]==0],
        df_discretized[df_discretized["Time"]==1],
        df_discretized[df_discretized["Time"]==2],
        df_discretized[df_discretized["Time"]==3],
        df_discretized[df_discretized["Time"]==4]
    ]

    for df_i in range(len(dataframes)):
        dataframes[df_i] = dataframes[df_i].drop(columns=["Sample Name", "Case", "Treatment", "Time"]).reset_index(drop=True)
        dataframes_disc[df_i] = dataframes_disc[df_i].drop(columns=["Sample Name", "Case", "Treatment", "Time"]).reset_index(drop=True)

        # Store real valued datasets
        dataframes[df_i].to_csv(curr_dir+"\\glycine_output\\real_"+str(df_i)+".csv")

        # discretized
        dataframes_disc[df_i].to_csv(curr_dir+"\\glycine_output\\discretized_"+str(df_i)+".csv")


def transform_glycine_diogo(files):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i])
        dataframes.append(df.drop(columns=[df.columns[0], df.columns[-1]]).T)

    output = open(curr_dir+"\\glycine_output\\glycine_transformed_diogo.tab", "w")
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

glycine_discretized_files = [
        curr_dir+"\\process_datasets\\glycine_output\\discretized_0.csv",
        curr_dir+"\\process_datasets\\glycine_output\\discretized_1.csv",
        curr_dir+"\\process_datasets\\glycine_output\\discretized_2.csv",
        curr_dir+"\\process_datasets\\glycine_output\\discretized_3.csv",
        curr_dir+"\\process_datasets\\glycine_output\\discretized_4.csv",
        curr_dir+"\\process_datasets\\glycine_output\\discretized_5.csv"
    ]

if __name__ == "__main__":
    process_glycine(glycine_file)

    transform_glycine_diogo(glycine_discretized_files)

