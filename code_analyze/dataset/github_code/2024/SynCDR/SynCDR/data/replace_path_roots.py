import os
import sys
sys.path.append(os.path.abspath('.'))

if __name__ == '__main__':
    dataset = 'domainnet'
    domains = ['clipart', 'painting', 'sketch']
    try:
        data_root_path = sys.argv[1]
        if data_root_path[-1] != '/':
            data_root_path += '/'
    except:
        raise Exception('Please provide the data root path as an argument')
    
    for d in domains:
        print('Domain:', d)
        for split in ['train', 'test', 'val', 'train_cls_disjoint1', 'train_cls_disjoint2']:
            print('Split:', split)
            filelist = f'{dataset}/{d}_{split}.txt'
            with open(filelist, 'r') as f:
                lines = f.read().splitlines()
            with open(filelist, 'w') as f:
                for line in lines:
                    f.write(line.replace('/home/ubuntu/data/', data_root_path) + '\n')
    
