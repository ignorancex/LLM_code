import matplotlib.pyplot as plt
import numpy as np

def split_lines_in_file(file_path, separator):
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
        
        # Split each line by the separator and store the result
        split_lines = [line.strip().split(separator) for line in lines]
        
        return split_lines
    
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return []
    
def extract(lst, index,default=0.0):
    ret = []
    for l in lst:
        try:
            num = float(l[index])
            ret.append(num)
        except:
            ret.append(default)

    return np.array(ret)
    
directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/act_inf_logs/ex8-precisions/k/"
filename = "combined.txt"

x_ticks = [0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3,0.5]
x_ticks = [x for x in map(str,x_ticks)]
data = split_lines_in_file(directory+"data/"+filename," ")
width = 0.8

# Red reach error
title = "Reach error (red ball)"
values = extract(data,0,42)

plt.bar(x_ticks, values, color ='salmon', width=width)

plt.subplots_adjust(left=0.08,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue reach error
title = "Reach error (blue ball)"
values = extract(data,1,42)

plt.bar(x_ticks, values, color ='navy', width = width)

plt.subplots_adjust(left=0.08,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Red reach time
title = "Reach time (red ball)"
values = extract(data,2,300)

plt.bar(x_ticks, values, color ='coral', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue reach time
title = "Reach time (blue ball)"
values = extract(data,3,300)

plt.bar(x_ticks, values, color ='darkslategrey', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Red stabiity
title = "Reach stability (red ball)"
values = extract(data,4,42)

plt.bar(x_ticks, values, color ='burlywood', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue stabiity
title = "Reach stability (blue ball)"
values = extract(data,5,42)

plt.bar(x_ticks, values, color ='lightsteelblue', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Red perception errr
title = "Perception error (red ball)"
values = extract(data,6,42)

plt.bar(x_ticks, values, color ='plum', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue perception errr
title = "Perception error (blue ball)"
values = extract(data,7,42)

plt.bar(x_ticks, values, color ='indigo', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Red perception errr
title = "Perception stability (red ball)"
values = extract(data,8,42)

plt.bar(x_ticks, values, color ='olivedrab', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue perception errr
title = "Perception stability (blue ball)"
values = extract(data,9,42)

plt.bar(x_ticks, values, color ='orange', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Blue angle std
title = "Object permanence direction deviation"
values = extract(data,10,42)

plt.bar(x_ticks, values, color ='seagreen', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()

# Object permanence
title = "Object permanence metric"
values = extract(data,11,42)

plt.bar(x_ticks, values, color ='yellowgreen', width = width)

plt.subplots_adjust(left=0.1,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
plt.xticks(x_ticks)
plt.xlabel("Value")
plt.ylabel(title)
plt.savefig(directory+"graphs/"+title)
plt.show()