# generate list for evaluation

import os
import argparse

DATA_ROOT = '../../../../../dataset/CelebAMask-HQ'

# path to the txt file for original celebA partition
CELEBA_PARITION = './list_eval_partition.txt'

SAVE_ROOT = '../../../../../dataset/CelebAMask-HQ/split_list'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',default=DATA_ROOT, help='Data root path of CelebAMask-HQ')
parser.add_argument('--celeba_partition', default=CELEBA_PARITION,
                    help='The partition file of celebA dataset')
parser.add_argument('--save_root', default=SAVE_ROOT,
                    help='The folder to save lists')

args = parser.parse_args()


def add_to_writer(writer, new_id, data_root, as_id=False):
    if as_id:
        content = new_id + '\n'
    else:
        image_root = 'CelebA-HQ-img-resize'
        label_root = 'CelebAMask-HQ-mask'

        img_path = os.path.join(data_root, image_root, new_id+'.jpg')
        assert os.path.exists(img_path), 'Invalid image path: '+img_path

        label_path = os.path.join(data_root, label_root, new_id+'.png')
        assert os.path.exists(label_path), 'Invalid image path: ' + label_path

        content = os.path.join('/'+image_root, new_id+'.jpg')+ ' ' + \
                  os.path.join('/'+label_root, new_id+'.png') + '\n'

    writer.write(content)


def main():

    data_root = args.data_root.strip()
    save_root = args.save_root.strip()
    save_root_id = os.path.join(save_root, 'id')
    celeba_partion_file = args.celeba_partition.strip()
    mapping_file = os.path.join(data_root, 'CelebA-HQ-to-CelebA-mapping.txt')

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if not os.path.exists(save_root_id):
        os.makedirs(save_root_id)

    train_writer = open(os.path.join(save_root, 'train.txt'), 'w')
    train_id_writer = open(os.path.join(save_root_id, 'train.txt'), 'w')

    val_writer = open(os.path.join(save_root,'val.txt'), 'w')
    val_id_writer = open(os.path.join(save_root_id, 'val.txt'), 'w')

    trainval_writer = open(os.path.join(save_root,'trainval.txt'), 'w')
    trainval_id_writer = open(os.path.join(save_root_id, 'trainval.txt'), 'w')

    test_writer = open(os.path.join(save_root,'test.txt'), 'w')
    test_id_writer = open(os.path.join(save_root_id, 'test.txt'), 'w')

    with open(celeba_partion_file,'r') as rd:
        lines = rd.readlines()
        parti_lines = [d.strip() for d in lines]
        train_set = []
        val_set = []
        test_set = []
        for ln in parti_lines:
            this_name, this_parti = ln.split()
            if this_parti == '0':
                train_set.append(this_name)
            elif this_parti == '1':
                val_set.append(this_name)
            elif this_parti == '2':
                test_set.append(this_name)
            else:
                raise ValueError('Invalid patition number at line: {}'.format(ln))

    with open(mapping_file, 'r') as rd:
        lines = rd.readlines()
        mapping_lines = [d.strip() for d in lines]
        mapping_lines = mapping_lines[1:]


    print('Generating train/val/test list for CelebAMask.\nResult saving root: {}'.format(save_root))

    for line in mapping_lines:
        new_id, _, ori_name = line.split()

        if ori_name in train_set:
            print('New id: {}, original name: {}, split: train'.format(new_id,ori_name))

            add_to_writer(train_writer, new_id, data_root, as_id=False)
            add_to_writer(train_id_writer, new_id, data_root, as_id=True)

            add_to_writer(trainval_writer, new_id, data_root, as_id=False)
            add_to_writer(trainval_id_writer, new_id, data_root, as_id=True)

        elif ori_name in val_set:
            print('New id: {}, original name: {}, split: val'.format(new_id, ori_name))

            add_to_writer(val_writer, new_id, data_root, as_id=False)
            add_to_writer(val_id_writer, new_id, data_root, as_id=True)

            add_to_writer(trainval_writer, new_id, data_root, as_id=False)
            add_to_writer(trainval_id_writer, new_id, data_root, as_id=True)

        elif ori_name in test_set:
            print('New id: {}, original name: {}, split: test'.format(new_id, ori_name))

            add_to_writer(test_writer, new_id, data_root, as_id=False)
            add_to_writer(test_id_writer, new_id, data_root, as_id=True)

        else:
            raise ValueError('Name: {} not found in all three sets. Examine bugs.'.format(ori_name))

    train_writer.close()
    train_id_writer.close()

    trainval_writer.close()
    trainval_id_writer.close()

    val_writer.close()
    val_id_writer.close()

    test_writer.close()
    test_id_writer.close()


if __name__=="__main__":
    main()