import matplotlib.pyplot as plt
import json

## Filenames
json_file_tags = "hist_valid_tags.txt"
json_file_set_indices = "hist_index_skews.txt"

## Folders
folder_list = ["VariantB_SAE_wGLE_Bugs", "VariantB_SAE_BugFix_AES", "VariantB_SAE_BugFix_PRINCE"]

## Distribution of Valid Tags
keys = []
values = []
for folder in folder_list[:2] :
    json_file = folder+"/"+json_file_tags
    with open(json_file, "r") as file:
        contents = file.readlines()
        keys.append(json.loads(contents[0].strip()))
        values.append(json.loads(contents[1].strip()))        

fig, axs = plt.subplots(1, 2, figsize=(10, 2.7))
axs[0].bar(keys[0], values[0], alpha=1,color='blue', edgecolor='black')
axs[0].set_yscale('log')
axs[0].set_xlabel('Number of Valid Tags Per Set')
axs[0].set_ylabel('Count of Sets (out of 32000)')
axs[0].set_title('(a) Original Code')
axs[0].set_xticks(range(15))
#axs[0].set_ylim(0,240)

axs[1].bar(keys[1], values[1], alpha=1,color='blue', edgecolor='black')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of Valid Tags Per Set')
axs[1].set_ylabel('Count of Sets (out of 32000)')
axs[1].set_title('(b) After Bug-Fix')
axs[1].set_xticks(range(15))
#axs[0].set_ylim(0,240)
fig.suptitle('Set-Occupancy in Tag-Store at Initialization',weight='bold')

plt.tight_layout()
plt.savefig("dist_valid_tags.pdf", )
plt.cla()
plt.clf()
    
## Distribution of Set-Indices in a Skew
bins = []
n_sorted = []
means = []
std_dev = []
for folder in folder_list :
    json_file = folder+"/"+json_file_set_indices
    with open(json_file, "r") as file:
        contents = file.readlines()
        bins.append(json.loads(contents[0].strip()))
        n_sorted.append(json.loads(contents[1].strip()))
        means.append(json.loads(contents[2].strip()))
        std_dev.append(json.loads(contents[3].strip()))

    
## print mean and std dev
print("Config \t\t\t\t Mean \t\t Std-Dev")
fig, axs = plt.subplots(1, 3, figsize=(9, 3))

## plot
for id,folder in enumerate(folder_list):
    if(id == 0):
        config = "(a) Original Indices       "
    elif(id == 1):
        config = "(b) Bug-Fix with AES Cipher"
    elif(id == 2):
        config = "(c) Bug-Fix with PRINCE Cipher"

    # print mean, std-dev
    print(config + "\t",format(float(means[id]),'.2f'),"\t\t",format(float(std_dev[id]),'.2f'))

    # plot
    axs[id].bar(bins[id], n_sorted[id], alpha=1,color='blue', edgecolor='black')
    axs[id].set_title(config)
    axs[id].set_xlabel('Indices (Sorted)')
    axs[id].set_ylabel('Count (out of 1 million)')
    axs[id].set_ylim(0,240)

## finalize plot
fig.suptitle('Distribution of Set-Index for 1 Million Addresses',weight='bold')
plt.tight_layout()
plt.savefig("dist_set_index.pdf", )
plt.cla()
plt.clf()
