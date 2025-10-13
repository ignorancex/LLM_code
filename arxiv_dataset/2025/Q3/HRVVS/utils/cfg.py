import argparse

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs', help='tensorboard log and model saved directory name')
    parser.add_argument('--gpu_device', type=int, default=0, help='gpu device index')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--data_path', type=str, default='../../../dataset/Hepa-Seg/history/vein', help='data path root')
    parser.add_argument('--epoch', type=int, default=80, help='max epoch number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--val_freq', type=int, default=1, help='val frequency')
    parser.add_argument('--image_size', type=int, default=1024, help='image size')
    parser.add_argument('--dataset', type=str, default='vein', help='name of dataset')
    opt = parser.parse_args()
    return opt