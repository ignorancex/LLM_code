import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

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
    sorted_data = sorted_data[sorted_data[:, 1] < 500] # < 500

    # Cut if i-th larger than x
    sorted_data = sorted_data[sorted_data[:, i] < max_dist]

    # Extract x and y values
    y = sorted_data[:, 0]
    smooth_y = gaussian_filter1d(y, sigma=2)
    z = sorted_data[:, 1]
    smooth_z = gaussian_filter1d(z, sigma=2)
    x = sorted_data[:, i]

    return x,smooth_y,smooth_z, np.mean(z)
    

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
directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/repo/AIF---visual-attention/act_inf_logs/experiments/action/"
file_paths = ["log_500_10_0_0_1000_attn_comp.csv","log_500_10_0_0_1000_dmu_lkh.csv"]
labels = ["Bottom-up action","Top-down action"]
colormap = cm.get_cmap("plasma")

# Sample 4 evenly spaced colors
colors = [colormap(i) for i in [0.1, 0.8, 0.9, 0.9]]
linestyles=["-","-.","-.",":"]

c = 3  # Change this to the column you want to sort by and plot
max_dist = 16
# Plot first column vs. i-th column
plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams.update({'font.size': 26})  # Change the number to the desired font size
plt.figure(figsize=(10, 5))
avgs=[]
datasets = []

for i in range(2):
    x,y,z,avg = read_and_plot_csv(directory+file_paths[i], c, max_dist)
    # x = x[y<300]
    # y = y[y<300]
    avgs.append(avg)
    datasets.append(x)
    datasets.append(z)
    plt.plot(x, z,color=colors[i], linestyle=linestyles[i], linewidth=4, label=labels[i])

save_csv("posner_action.csv",datasets)

print("Averages",avgs)
for i in range(2):
    plt.axhline(y=avgs[i], color=colors[i], linestyle='--', linewidth=4)

plt.ylabel('Reach Time (steps)')
plt.xlabel(f'Target distance from focus point (px)')
plt.legend()
# plt.title(f'Plot of First Column vs. Column {i}')
plt.grid(True)
plt.show()