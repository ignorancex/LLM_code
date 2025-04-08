import os
import numpy as np

def split_train_val(data, ratio_train=None):
    if os.path.isfile('%s/Train_validation_idx_%d.txt'%(data.chpath,data.n_conformer)):
        train_conf_idx = []; val_conf_idx = []
        with open('%s/Train_validation_idx_%d.txt'%(data.chpath,data.n_conformer),'r') as reader:
            lines = reader.readlines()
            train_conf_idx = [int(i) for i in lines[1].split('\t')]
            val_conf_idx = [int(i) for i in lines[3].split('\t')]
    
    else:
        solv2solus = {}
        for npzf in data.npzs:
            confID = npzf[-5]
            if confID == '0':
                id = os.path.basename(npzf)[:-5]
                solvent = int(id.split('_')[0])
                if solvent not in solv2solus:
                    solv2solus[solvent] = []
                solv2solus[solvent].append(id)
        if ratio_train > 1:
            print ('num_train should be 0~1')
            exit()

        
        train_idx = []; val_idx = []

        for ids in solv2solus.values():
            if len(ids) < 10:
                train_idx.extend(ids)
            else:
                ids = np.random.permutation(ids)
                trainN = int(len(ids)*ratio_train)
                train_idx.extend(ids[:trainN])
                val_idx.extend(ids[trainN:])

        train_conf_idx = []; val_conf_idx = []
        for npz in train_idx:
            for i in range(data.n_conformer):
                solv_solu_conf = '%s/%s%s.npz'%(data.dbpath,npz,i)
                if os.path.isfile(solv_solu_conf):
                    train_conf_idx.append(data.npzs.index(solv_solu_conf))
                else:
                    break
        
        for npz in val_idx:
            for i in range(data.n_conformer):
                solv_solu_conf = '%s/%s%s.npz'%(data.dbpath,npz,i)
                if os.path.isfile(solv_solu_conf):
                    val_conf_idx.append(data.npzs.index(solv_solu_conf))
                else:
                    break
        train_conf_idx = np.random.permutation(train_conf_idx)


        with open('%s/Train_validation_idx_%s.txt'%(data.chpath,data.n_conformer),'w') as out:
            print ('save training validation index')
            out.write('Training\n')
            for i in train_conf_idx[:-1]:
                out.write('%d\t'%i)
            out.write('%d\n'%train_conf_idx[-1])

            out.write('Validation\n')
            for i in val_conf_idx[:-1]:
                out.write('%d\t'%i)
            out.write('%d\n'%val_conf_idx[-1])

    train = data.create_subset(train_conf_idx)
    val = data.create_subset(val_conf_idx)

    return train, val

def train_val_test_split(
    data,
    num_train=None,
    num_val=None,
    num_test=None,
    checkpoint_path=None
):
    """
    Splits the dataset into train/validation/test splits, writes split to
    an npz file and returns subsets. Either the sizes of training and
    validation split or an existing split file with split indices have to
    be supplied. The remaining data will be used in the test dataset.

    Args:
        num_train (int): number of training examples
        num_val (int): number of validation examples
        
    Returns:
        schnetpack.data.AtomsData: training dataset
        schnetpack.data.AtomsData: validation dataset
        schnetpack.data.AtomsData: test dataset

    """
    
    if num_train is None or num_val is None or num_test is None:
        raise ValueError(
            "You have to supply either split sizes (num_train, num_val, num_test) or an npz file with splits."
        )

    assert num_train + num_val + num_test<= len(data), "Dataset is smaller than num_train + num_val + num_test !"

    num_train = num_train if num_train > 1 else num_train * len(data)
    num_val = num_val if num_val > 1 else num_val * len(data)
    num_test = num_test if num_test > 1 else num_test * len(data)
    num_train = int(num_train)
    num_val = int(num_val)
    num_test = int(num_test)
    
   
    idx = np.random.permutation(len(data))
    #idx = np.arange(len(data))
    print ('dataNo: ',len(idx))
    train_idx = idx[:num_train].tolist()
    val_idx = idx[num_train:num_train+num_test].tolist()
    test_idx = idx[num_train+num_test:].tolist()
    
    
    if os.path.isfile('%s/Train_validation_test_idx.txt'%checkpoint_path):
        print ('Train_validation_test_idx is exist')
        #exit()
    
    with open('%s/Train_validation_test_idx.txt'%checkpoint_path,'w') as out:
        print ('save training validation test index')
        out.write('Training\n')
        for i in train_idx[:-1]:
           out.write('%d\t'%i)
        out.write('%d\n'%train_idx[-1])

        out.write('Validation\n')
        for i in val_idx[:-1]:
           out.write('%d\t'%i)
        out.write('%d\n'%val_idx[-1]) 

        out.write('Test\n')
        for i in test_idx[:-1]:
           out.write('%d\t'%i)
        out.write('%d\n'%test_idx[-1]) 
    '''
    train_idx = []
    val_idx = []
    with open('%s/Train_validation_idx.txt'%checkpoint_path,'r') as input:
        print ('load training validation index')
        lines = input.readlines()
        train_idx = [int(i) for i in lines[1].split('\t')]
        val_idx = [int(i) for i in lines[3].split('\t')]
    
    '''
    train = data.create_subset(train_idx)
    val = data.create_subset(val_idx)
    test = data.create_subset(test_idx)
    return train, val, test
