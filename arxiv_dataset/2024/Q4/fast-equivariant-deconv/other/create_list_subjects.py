import os
import numpy as np

list_sub = os.listdir('/scratch/age261/hpc_wumin_media')
a = []
for sub in list_sub:
    if sub.isdigit():
        a.append(sub)
np.random.shuffle(a)
b = []
for sub in a[:65]:
    b.append(f'/scratch/age261/hpc_wumin_media/{sub}')
os.makedirs('/scratch/age261/hpc_wumin_media/train_split', exist_ok=True)
np.savetxt('/scratch/age261/hpc_wumin_media/train_split/list_subjects.txt', b, fmt="%s")
print(b)

b = []
for sub in a[65:85]:
    b.append(f'/scratch/age261/hpc_wumin_media/{sub}')
os.makedirs('/scratch/age261/hpc_wumin_media/test_split', exist_ok=True)
np.savetxt('/scratch/age261/hpc_wumin_media/test_split/list_subjects.txt', b, fmt="%s")
print(b)

b = []
for sub in a[85:]:
    b.append(f'/scratch/age261/hpc_wumin_media/{sub}')
os.makedirs('/scratch/age261/hpc_wumin_media/val_split', exist_ok=True)
np.savetxt('/scratch/age261/hpc_wumin_media/val_split/list_subjects.txt', b, fmt="%s")
print(b)