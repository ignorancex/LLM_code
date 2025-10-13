import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def read_and_plot_csv(file_path, i, max_dist=12):
    # Read the CSV file, assuming it has a header and delimiter is ','
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    
    # Ensure the column index is valid
    if i >= data.shape[1] or i < 0:
        raise ValueError(f"Column index {i} is out of range.")
    
    # data = data[np.random.choice(data.shape[0], size=199, replace=False)]
    # Sort by the i-th column
    sorted_data = data[np.argsort(data[:, i])] # :100 / :50

    # Remove rows where the first column has the value 1000
    sorted_data = sorted_data[sorted_data[:, 0] < 1000] # < 500

    # Cut if i-th larger than x
    sorted_data = sorted_data[sorted_data[:, i] < max_dist]

    # Extract x and y values
    y = sorted_data[:, 0]
    smooth_y = gaussian_filter1d(y, sigma=2)
    x = sorted_data[:, i]

    return x,smooth_y, np.mean(y)
    
def save_csv(filename,datasets):
    padding = np.max(list(map(np.size,datasets)))
    with open(filename,'w') as f:
        for i in range(padding):
            line = ""
            for d in datasets:
                if i>=len(d):
                    line+="NaN,"
                else:
                    line+=str(d[i])+","
            line = line[:-1]+"\n"
            f.write(line)


# Example usage
coa = "100"
directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/repo/AIF---visual-attention/act_inf_logs/experiments/coa"+coa+"/"
file_paths = ["log_200_10_50_"+coa+"_1000_True_True_False.csv","log_200_10_50_"+coa+"_1000_True_False_False.csv","log_200_10_50_"+coa+"_1000_False_True_False.csv","log_200_10_50_"+coa+"_1000_False_False_False.csv"]
labels = ["Endogenous - Valid","Endogenous - Invalid","Exogenous - Valid","Exogenous - Invalid"]
# colors = [(31/255, 119/255, 180/255),  # Original blue (#1f77b4)
#           (255/255, 127/255, 14/255),  # Original orange (#ff7f0e)
#           (44/255, 160/255, 44/255),  # Original green (#2ca02c)
#           (214/255, 39/255, 40/255)]  # Original red (#d62728)

# lighter_colors = [(123/255, 182/255, 221/255),  # Lighter blue (#85c0e7)
#                   (245/255, 173/255, 113/255),  # Lighter orange (#ffb77b)
#                   (143/255, 220/255, 143/255),  # Lighter green (#99e699)
#                   (234/255, 167/255, 167/255)]  # Lighter red (#f4b1b1)

colormap = cm.get_cmap("plasma")

# Sample 4 evenly spaced colors
colors = [colormap(i) for i in [0.1, 0.3, 0.6, 0.9]]

# # Normal grayscale colors
# colors = [(0.0, 0.0, 0.0),  # Dark gray
#           (0.5, 0.5, 0.5),  # Medium gray
#           (0.3, 0.3, 0.3),  # Medium-dark gray
#           (0.75, 0.75, 0.75)]  # Light gray

linestyles=["-","--","-.",":"]

c = 3  # Change this to the column you want to sort by and plot
max_dist = 16
# Plot first column vs. i-th column
plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams.update({'font.size': 24})  # Change the number to the desired font size
plt.figure(figsize=(10, 5))
datasets = []
avgs=[]
for i in range(4):
    x,y,avg = read_and_plot_csv(directory+file_paths[i], c, max_dist)
    x = x[y<300]
    y = y[y<300]
    avgs.append(avg)
    datasets.append(x)
    datasets.append(y)
    plt.plot(x, y,color=colors[i], linestyle=linestyles[i], linewidth=5, label=labels[i])

for i in range(4):
    plt.axhline(y=avgs[i], color=colors[i], linestyle=linestyles[i], linewidth=5)

print("Averages:",avgs)
save_csv("posner_dist.csv",datasets)


plt.ylabel('Reaction Time (steps)')
plt.xlabel(f'Target distance from focus point (px)')
plt.legend()
# plt.title(f'Plot of First Column vs. Column {i}')
plt.grid(True)
plt.show()

# Plot COA averages
ctoa_x = [0,100,200,300,400,500,600]
endo_val=[]
endo_inval = []
exo_val = []
exo_inval = []
lists = [endo_val,endo_inval,exo_val,exo_inval]

for ctoa in ctoa_x:
    coa = str(ctoa)
    directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/repo/AIF---visual-attention/act_inf_logs/experiments/coa"+coa+"/"
    file_paths = ["log_200_10_50_"+coa+"_1000_True_True_False.csv","log_200_10_50_"+coa+"_1000_True_False_False.csv","log_200_10_50_"+coa+"_1000_False_True_False.csv","log_200_10_50_"+coa+"_1000_False_False_False.csv"]
    for i in range(4):
        x,y,avg = read_and_plot_csv(directory+file_paths[i], c, max_dist)
        lists[i].append(avg)

plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams.update({'font.size': 24})  # Change the number to the desired font size
plt.figure(figsize=(10, 5))
datasets=[ctoa_x]
for i in range(4):
    datasets.append(lists[i])
    plt.plot(ctoa_x, lists[i],color=colors[i], linestyle=linestyles[i], linewidth=5, label=labels[i])

save_csv("posner_ctoa.csv",datasets)

plt.ylabel('Reaction Time (steps)')
plt.xlabel('Cue-Target Onset Asynchrony (steps)')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()