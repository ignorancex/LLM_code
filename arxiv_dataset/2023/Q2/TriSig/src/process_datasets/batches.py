import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import DI2 as di2
import numpy as np
import math
import os.path as path

curr_dir = os.getcwd()
data_folder = path.abspath(path.join(__file__, "../../.."))+"\\data\\batches\\"


def is_number(n):
    is_number = True
    try:
        num = float(n)
        # check for "nan" floats
        if math.isnan(num):
            return False
        return True   # or use `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number
    

def read_batchs(files, transpose=True):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i]).reset_index(drop=True) #
        df = df.drop(columns=[df.columns[0]]) #.to_numpy()
        df.columns = list(range(0,len(df.columns)))
        dataframes.append(df.T)

    final_dataframe = [[[0 for z in range(len(dataframes))] for j in range(len(dataframes[0][dataframes[0].columns[0]]))] for i in range(len(dataframes[0].columns)-1)]

    for batch_i in range(0, len(dataframes)):
        for time_point_i in range(0, len(dataframes[0].columns)-1):
            for variable_i in range(0, len(dataframes[0][dataframes[0].columns[0]])):
                final_dataframe[time_point_i][variable_i][batch_i] = dataframes[batch_i][dataframes[0].columns[time_point_i]].iat[variable_i]

    f_d = []
    for time_point_i in range(len(final_dataframe)):
        f_d.append(pd.DataFrame(final_dataframe[time_point_i]).T if transpose else pd.DataFrame(final_dataframe[time_point_i]))

    return f_d


def moving_average(time_series, window):
    time_series = time_series.reset_index(drop=True)
    for i in range(0, len(time_series)):
        if not is_number(time_series[i]) or pd.isna(time_series[i]):
            index = []
            # Get values before
            for before in range(i-window, i):
                if before < 0:
                    continue
                else:
                    index.append(float(time_series[before]))
            # Get values after
            for after in range(i+1, i+window+1):
                if after >= len(time_series):
                    break
                else:
                    if not is_number(time_series[after]) or pd.isna(time_series[after]):
                        continue
                    else:
                        index.append(float(time_series[after]))
            # value to be inputted
            value = 0
            if len(index) == 0:
                for val in range(i+1, len(time_series)):
                    if is_number(time_series[val]) or not pd.isna(time_series[val]):
                        value = time_series[val]
                        break
            else:
                value = sum(index)/len(index)
            # change time series value at i
            time_series.iat[i] = value

    return time_series


def average_values_in_bins(df, bins):
    binned_averages = {}
    for column in df:
        values = list(df[column].values)
        part_len = len(values)//bins
        bin_averages = [values[i:i + part_len] for i in range(0, len(values), part_len)]
        for b in range(0, len(bin_averages)):
            bin_averages[b] = np.mean(bin_averages[b])
        binned_averages[column] = bin_averages
    return pd.DataFrame(binned_averages)


###################################### Batches data ############################################

original_batch_file = data_folder+"\\batches\\100_Batches_IndPenSim_V3.csv"


def process_batch_data(file):
    df = pd.read_csv(file).iloc[:,0:36]
    df = df.drop(columns=[
        " 1-Raman spec recorded",
        "1- No Raman spec",
        '0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)',
        'Fault reference(Fault_ref:Fault ref)',
        'Ammonia shots(NH3_shots:kgs)', #Doesn't vary
        'Carbon evolution rate(CER:g/h)', #Doesn't vary
        'Oxygen Uptake Rate(OUR:(g min^{-1}))', #Doesn't vary
        'Generated heat(Q:kJ)', #Doesn't vary
        'Vessel Weight(Wt:Kg)', #Doesn't vary
        'Vessel Volume(V:L)', #Doesn't vary
        'Dumped broth flow(Fremoved:L/h)', #Doesn't vary
        'Agitator RPM(RPM:RPM)', #Doesn't vary
        'Heating water flow rate(Fh:L/h)', # Doesn't vary
        'Time (h)',
        "PAA concentration offline(PAA_offline:PAA (g L^{-1}))", # Missing values
        "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))", # Missing values
        "Offline Penicillin concentration(P_offline:P(g L^{-1}))", # Missing values
        "Offline Biomass concentratio(X_offline:X(g L^{-1}))", #missing
        "Viscosity(Viscosity_offline:centPoise)", #Missing
    ])


    dataframes = []

    print("Normalize Min_max")
    for col in df.columns:
        if col not in ["2-PAT control(PAT_ref:PAT ref)"]:
            df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

    for batch_i in range(1,101):
        dataframes.append(df[df["2-PAT control(PAT_ref:PAT ref)"] == batch_i].drop(columns=["2-PAT control(PAT_ref:PAT ref)"]).reset_index(drop=True))

    print("Doing average value in bins")
    for df_i in range(len(dataframes)):
        dataframes[df_i] = average_values_in_bins(dataframes[df_i], 25).head(25)

        # Store real valued datasets
        dataframes[df_i].to_csv(curr_dir+"\\batches_output\\real_batches"+str(df_i)+".csv")

    #dataset to discretize
    combined_df = pd.concat(dataframes).reset_index(drop=True)
    for col in combined_df.columns:
        if col not in ["2-PAT control(PAT_ref:PAT ref)"]:
            combined_df[col] = di2.distribution_discretizer(combined_df[col].to_frame(), number_of_bins=7, cutoff_margin=0.0, single_column_discretization=True).iloc[:, 0]

    dataframes_discretized = np.array_split(combined_df, 100)

    for df_i in range(len(dataframes_discretized)):
        dataframes_discretized[df_i].to_csv(curr_dir+"\\batches_output\\discretized_batches"+str(df_i)+".csv")


def transform_batch_diogo(files):
    dataframes = []
    for file_i in range(len(files)):
        df = pd.read_csv(files[file_i])
        df = df.drop(columns=[df.columns[0]])
        df.columns = list(range(0,len(df.columns)))
        dataframes.append(df.T)

    final_dataframe = [[[0 for z in range(len(dataframes))] for j in range(len(dataframes[0][dataframes[0].columns[0]]))] for i in range(len(dataframes[0].columns)-1)]

    for batch_i in range(0, len(dataframes)):
        for time_point_i in range(0, len(dataframes[0].columns)-1):
            for variable_i in range(0, len(dataframes[0][dataframes[0].columns[0]])):
                final_dataframe[time_point_i][variable_i][batch_i] = dataframes[batch_i][dataframes[0].columns[time_point_i]].iat[variable_i]

    f_d = []
    for time_point_i in range(len(final_dataframe)):
        f_d.append(pd.DataFrame(final_dataframe[time_point_i]))

    output = open(curr_dir+"\\batches_output\\batches_transformed_diogo.tab", "w")
    output.write("Total Time:\t"+str(len(f_d))+"\n")
    output.write("Total Samples:\t"+str(len(f_d[0].columns))+"\n")
    output.write("Total Genes:\t"+str(len(f_d[0][f_d[0].columns[0]]))+"\n")

    for z_i in range(len(f_d)):
        output.write("Time\t"+str(z_i)+"\n")
        output.write("ID\tNAME")
        for col_i in range(len(f_d[z_i].columns)):
            output.write("\tS-"+str(col_i))
        output.write("\n")
        for x_i in range(len(f_d[z_i][f_d[z_i].columns[0]])):
            output.write(str(x_i)+"\tG-"+str(x_i))
            for y_i in range(len(f_d[z_i].columns)):
                output.write("\t"+str(f_d[z_i].iloc[x_i][f_d[z_i].columns[y_i]]))
            output.write("\t\n")

    output.close()

discretized_batch_files = []

for df_i in range(0,100):
    discretized_batch_files.append(curr_dir+"\\process_datasets\\batches_output\\discretized_batches"+str(df_i)+".csv")

if __name__ == "__main__":
    process_batch_data(original_batch_file)

    transform_batch_diogo(discretized_batch_files)
