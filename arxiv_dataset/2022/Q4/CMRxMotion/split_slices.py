import os
import glob

import numpy as np

def check_class(f, c1, c2, c3):
    flag = [False, False, False]
    for x in f:
        if x in c1:
            flag[0] = True
        if x in c2:
            flag[1] = True
        if x in c3:
            flag[2] = True
    return all(flag)

if __name__ == '__main__':

    while True:

        fs1 = [os.path.basename(x).split('-E')[-2] for x in glob.glob('./data_cls/lab_1/*')]
        fs2 = [os.path.basename(x).split('-E')[-2] for x in glob.glob('./data_cls/lab_2/*')]
        fs3 = [os.path.basename(x).split('-E')[-2] for x in glob.glob('./data_cls/lab_3/*')]

        fs1 = [x.split('-')[0] for x in sorted(fs1)]
        fs2 = [x.split('-')[0] for x in sorted(fs2)]
        fs3 = [x.split('-')[0] for x in sorted(fs3)]

        f = np.unique(sorted(fs1 + fs2 + fs3))
        f = np.unique([x.split('-')[0] for x in f])

        flag = True
        while flag:
            np.random.shuffle(f)
            f1 = f[0:4]
            f2 = f[4:8]
            f3 = f[8:12]
            f4 = f[12:16]
            f5 = f[16:20]
            flag = not all([check_class(f1, fs1, fs2, fs3),
                            check_class(f2, fs1, fs2, fs3),
                            check_class(f3, fs1, fs2, fs3),
                            check_class(f4, fs1, fs2, fs3),
                            check_class(f5, fs1, fs2, fs3)])

        folds = [f1, f2, f3, f4, f5]
            

        fs1 = sorted([x.replace('\\', '/') for x in glob.glob('./data_cls/lab_1/*')])
        fs2 = sorted([x.replace('\\', '/') for x in glob.glob('./data_cls/lab_2/*')])
        fs3 = sorted([x.replace('\\', '/') for x in glob.glob('./data_cls/lab_3/*')])


        print()

        data_dict = {}
        for i, fold in enumerate(folds):
            sub_c1 = []
            sub_c2 = []
            sub_c3 = []
            for k in fold:
                sub_c1 += [x for x in fs1 if k in x]
                sub_c2 += [x for x in fs2 if k in x]
                sub_c3 += [x for x in fs3 if k in x]
            data_dict.update({f'fold_{i+1}_c1': sub_c1,
                            f'fold_{i+1}_c2': sub_c2,
                            f'fold_{i+1}_c3': sub_c3})


        lc1s = []
        lc2s = []
        lc3s = []
        for i in range(1, 6):
            lc1 = len(data_dict[f'fold_{i}_c1'])
            lc2 = len(data_dict[f'fold_{i}_c2'])
            lc3 = len(data_dict[f'fold_{i}_c3'])
            print(f'fold {i}: {lc1}-{lc2}-{lc3}')
            lc1s.append(lc1)
            lc2s.append(lc2)
            lc3s.append(lc3)
        
        if all([x>10 for x in lc1s]) and all([x>10 for x in lc2s]) and all([x>2 for x in lc3s]):
            break

    np.save('./data_cls/5folds.npy', data_dict)
