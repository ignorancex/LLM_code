import matplotlib.pyplot as plt
import numpy as np

def calculate_row_wise_angles(matrix1, matrix2):
    # Step 1: Compute the dot products row-wise
    dot_products = np.einsum('ij,ij->i', matrix1, matrix2)
    
    # Step 2: Compute the magnitudes of the vectors row-wise
    magnitudes1 = np.linalg.norm(matrix1, axis=1)
    magnitudes2 = np.linalg.norm(matrix2, axis=1)
    
    # Step 3: Compute the cosines of the angles
    cos_angles = dot_products / (magnitudes1 * magnitudes2)
    
    # Step 4: Compute the angles in radians
    angles = np.arccos(cos_angles)
    
    return angles

def remove_first_occurrence(input_string, char_to_remove):
    # Find the index of the first occurrence of the character
    index = input_string.find(char_to_remove)
    
    # If the character is found, remove it
    if index != -1:
        input_string = input_string[:index] + input_string[index+1:]
    
    return input_string

directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/act_inf_logs/ex1-focus_on_static/"
name = "ex1_red-out"
log_name = directory+name+".csv"

data = np.genfromtxt(log_name, delimiter=',')
end = -1#300
obj_perm = True
attn_change = False
period = 300

needs = data[:end,:2]
targets = data[:end,2:2+2*2]
projections = data[:end,2+2*2:]

red_intentions = targets[:,:2] - 16
red_projections = projections[:,:2] - 16

blue_intentions0 = targets[:,2:4] -16
blue_projections0 = projections[:,2:4] -16

red_intentions = np.linalg.norm(red_intentions,axis=1)
red_projections = np.linalg.norm(red_projections,axis=1)
blue_intentions = np.linalg.norm(blue_intentions0,axis=1)
blue_projections = np.linalg.norm(blue_projections0,axis=1)
blue_dist = np.abs(blue_projections-blue_intentions)
blue_recip = 1/blue_dist
max_tick = np.max((red_intentions,red_projections,blue_intentions,blue_projections))

plt.plot(red_projections,color="red",label="Red ball projection",linewidth=2)
plt.plot(red_intentions,color="pink",label="Red ball perception",linewidth=2)
plt.plot(blue_projections,color="blue",label="Blue ball projection",linewidth=2)
plt.plot(blue_intentions,color="skyblue",label="Blue ball perception",linewidth=2)


if attn_change:
    # find change indices
    colors = ["red","blue"]
    indices = list(range(0,len(needs),period))
    indices.append(len(needs))
    print(indices)
    color_ind = np.where(needs[0]==1)[0][0]
    for i in range(len(indices)-1):
        plt.axvspan(indices[i], indices[i+1], facecolor=colors[color_ind], alpha=0.1)
        color_ind+=1
        color_ind%=2


plt.subplots_adjust(left=0.08,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)


plt.yticks(np.arange(0, max_tick+1, max_tick//10))
plt.xlabel("Steps")
plt.ylabel("Pixel distance")
plt.legend()

plt.savefig(directory+"graphs/"+remove_first_occurrence(name,"."))
plt.show()


# Blue object permanence plot
if obj_perm:
    plt.plot(blue_dist,color="teal",label="Blue perception difference",linewidth=2)

    if attn_change:
        # find change indices
        colors = ["red","blue"]
        indices = list(range(0,len(needs),period))
        indices.append(len(needs))
        print(indices)
        color_ind = np.where(needs[0]==1)[0][0]
        for i in range(len(indices)-1):
            plt.axvspan(indices[i], indices[i+1], facecolor=colors[color_ind], alpha=0.1)
            color_ind+=1
            color_ind%=2
    else:
        try:
            out_of_fov = np.where(blue_projections >= 16)[0][0]
            plt.axvline(x = out_of_fov, color = 'navy', label = 'Blue ball out of sight',linestyle="dashed",linewidth=2)
            red_reached = np.where(red_projections <= 1.5)[0][0]
            plt.axvline(x = red_reached, color = 'firebrick', label = 'Red ball reached',linestyle="dashed",linewidth=2)
        except:
            pass

    # plt.yticks(np.arange(0, max_tick+1, max_tick//10))
    plt.subplots_adjust(left=0.08,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)
    plt.xlabel("Steps")
    plt.ylabel("Pixel difference")
    plt.legend()

    plt.savefig(directory+"graphs/"+remove_first_occurrence(name,".")+"_perm")
    plt.show()

# Performance
lineout = ""

# Error at end of trial
red_reach = red_projections[-1]
print("Red reach error=",red_reach)
lineout+=" "+str(red_reach)
blue_reach = blue_projections[-1]
print("Blue reach error=",blue_reach)
lineout+=" "+str(blue_reach)

# Reach time
red_time = None
try:
    red_time = np.where(red_projections <= 1.5)[0][0]
except:
    pass
print("Red reach time=",red_time)
lineout+=" "+str(red_time)
blue_time = None
try:
    blue_time = np.where(blue_projections <= 1.5)[0][0]
except:
    pass
print("Blue reach time=",blue_time)
lineout+=" "+str(blue_time)

# Reach stability
red_stability = None
if red_time!=None: red_stability = np.std(red_projections[red_time:])
print("Red stability=",red_stability)
lineout+=" "+str(red_stability)
blue_stability = None
if blue_time!=None: blue_stability = np.std(blue_projections[blue_time:])
print("Blue stability=",blue_stability)
lineout+=" "+str(blue_stability)

# Peception error at end of trial
red_perception = np.abs(red_projections[-1]-red_intentions[-1])
print("Red perception error=",red_perception)
lineout+=" "+str(red_perception)
blue_perception = np.abs(blue_projections[-1]-blue_intentions[-1])
print("Blue perception error=",blue_perception)
lineout+=" "+str(blue_perception)

# Perception stability
red_correct_estimation = None
try:
    red_correct_estimation = np.where(np.abs(red_projections-red_intentions) <= 1)[0][0]
except:
    pass
red_perception_stability = None if red_correct_estimation==None else np.std(np.abs(red_projections-red_intentions)[red_correct_estimation:])
print("Red perception stability=",red_perception_stability)
lineout+=" "+str(red_perception_stability)

blue_correct_estimation = None
try:
    blue_correct_estimation = np.where(np.abs(blue_projections-blue_intentions) <= 1)[0][0]
except:
    pass
blue_perception_stability = None if blue_correct_estimation==None else np.std(np.abs(blue_projections-blue_intentions)[blue_correct_estimation:])
print("Blue perception stability=",blue_perception_stability)
lineout+=" "+str(blue_perception_stability)

# blue angle std
blue_angles = calculate_row_wise_angles(blue_projections0,blue_intentions0)
blue_angle_std = np.std(blue_angles)
print("Blue angle stability",blue_angle_std)
lineout+=" "+str(blue_angle_std)

# obj permanence metric
obj_perm_metric = np.mean(blue_recip)
print("Blue object permanence metric:",obj_perm_metric)
lineout+=" "+str(obj_perm_metric)

print(lineout)
with open(directory+"data/"+remove_first_occurrence(name,".")+".txt","w") as f:
    f.write(lineout[1:])
