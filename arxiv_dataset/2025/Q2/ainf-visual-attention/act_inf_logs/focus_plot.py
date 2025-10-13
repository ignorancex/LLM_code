import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

target_dist = 7.15
flip = [False,True,False,True]

def read_and_plot_csv(file_path,i):
    # Load data from CSV
    data = np.genfromtxt(file_path, delimiter=",")  # Adjust delimiter if needed

    # Extract coordinate pairs (assuming columns: x1, y1, x2, y2)
    x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Reference point
    ref_x, ref_y = 16, 16

    # Compute Euclidean distances
    dist1 = np.sqrt((x1 - ref_x) ** 2 + (y1 - ref_y) ** 2)
    dist2 = np.sqrt((x2 - ref_x) ** 2 + (y2 - ref_y) ** 2)

    if flip[i] == True:
        cue = 160
        x1_pc = x1[cue:]
        # print(x1_pc)
        x2_pc = x2[cue:]
        # print(np.where(x2_pc < 20)[0])
        p1 = np.where(x1_pc < 16)[0][0] + cue
        p2 = np.where(x2_pc < 16)[0][0] + cue
        dist1[p1:]=-dist1[p1:]
        dist2[p2:]=-dist2[p2:]

    return np.clip(dist1,-target_dist,target_dist), np.clip(dist2,-target_dist,target_dist)
    
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
directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/repo/AIF---visual-attention/act_inf_logs/experiments/"
file_paths = ["log_1_10_50_100_1000_True_True_False.csv","log_1_10_50_100_1000_True_False_False.csv","log_1_10_50_100_1000_False_True_False.csv","log_1_10_50_100_1000_False_False_False.csv"]
labels = ["Endogenous","Endogenous-Invalid", "Exogenous","Exogenous-Invalid"]
# colors = [(0.3, 0.3, 0.3),  # Medium-dark gray
#           (0.0, 0.0, 0.0),  # Dark gray
#           (0.75, 0.75, 0.75),  # Medium gray
#           (0.75, 0.75, 0.75)]  # Light gray

colormap = cm.get_cmap("plasma")

# Sample 4 evenly spaced colors
colors = [colormap(i) for i in [0.1, 0.3, 0.8, 0.9]]

linestyles=["-","--","-",":"]

# Plot first column vs. i-th column
plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams.update({'font.size': 26})  # Change the number to the desired font size
plt.figure(figsize=(10, 5))

plt.axvline(x=11, color="gray", linestyle='--', linewidth=2)
plt.axvline(x=61, color="gray", linestyle='--', linewidth=2)
plt.axvline(x=161, color="gray", linestyle='--', linewidth=2)

plt.axhline(y=target_dist, color=colormap(0.6), linestyle='--', linewidth=2)

datasets=[np.linspace(1,500,500)]

for i in [0,2]:
    x,y = read_and_plot_csv(directory+file_paths[i],i)
    # x = x[y<300]
    # y = y[y<300]
    datasets.append(x)
    datasets.append(y)
    plt.plot(x,color=colors[i], linestyle=linestyles[i], linewidth=5, label=labels[i])
    plt.plot(y,color=colors[i], linestyle="--", linewidth=5)

# plt.axhline(y=-target_dist, color="red", linestyle='--', linewidth=1)

save_csv("posner_target.csv",datasets)

plt.xlabel('Simulation Time (steps)')
plt.ylabel('Distance from center (px)')
plt.legend()
# plt.title(f'Plot of First Column vs. Column {i}')
plt.grid(True)
plt.show()