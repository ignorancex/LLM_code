from bnbad import *
import numpy as np


# Function to do insertion sort
def insertionSort(arr, t):
    
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        print(i)
        key = arr[i]
  
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >=0 and not t.better(key, arr[j]):
                arr[j+1] = arr[j]
                j -= 1
        arr[j+1] = key
    
with open ("../datasets/osp_complete.csv") as csv:
    data = csv.read()
    t = Tab().read(data)

new_rows = t.rows.copy()
insertionSort(new_rows, t)

indexes = []
for o_row in t.rows:
    for i, n_row in enumerate(new_rows):
        if o_row == n_row:
            indexes.append(i+1)
            print(i)
            break

idx = np.array(indexes)
np.savetxt("../datasets/osp_ranking.csv", idx, delimiter=",")

