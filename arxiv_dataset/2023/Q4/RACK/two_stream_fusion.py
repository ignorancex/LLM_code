import torch.nn.functional as F
import torch
import numpy as np
import pdb
parser = argparse.ArgumentParser(description='two-stream fusion of RACK')
parser.add_argument('--segments', dest='segments',
                  help='the number segments for a full video',
                  default=10, type=int)
parser.add_argument('--path_rgb_logits', dest='path_rgb_logits', # logits of Rack-RGB
                  help='path of the Rack-RGB's logits',
                  default=os.path.normpath("results/hmdb51/hmdb51_rgb_scores3.npy"), type=str)
               
parser.add_argument('--path_flow_logits', dest='path_flow_logits', # logits of Rack-flow
                  help='path of the Rack-flow's logits',
                  default=os.path.normpath("results/hmdb51/hmdb51_flow_scores3.npy"), type=str)
parser.add_argument('--telabel', dest='telabel', # test label
                  help='path label file's logits',
                  default=os.path.normpath("data/hmdb51/testlabel_rgb_split3.npy"), type=str)
                  

args = parser.parse_args()

print('args:', args)

telabel = np.load(args.telabel)  # (N,1) 
rgb_logits=np.load(args.path_rgb_logits) # (N,segments, nClass), N: samples, segments: segments of the full videos, nClass: number of classes
flow_logits=np.load(args.path_flow_logits) 
fuse_logits=rgb_logits + flow_logits # element-wise summation of the logits
fuse_logits=torch.from_numpy(fuse_logits)
test_accuracy=[]
for i in range(10):
        test_score = F.softmax(fuse_logits[:,i], dim=1)  # softmax layer
        pred_y = torch.max(test_score, 1)[1].data.numpy()  # predicted labels
        accuracy = float((pred_y == telabel).astype(int).sum()) / float(telabel.shape[0])
        test_accuracy.append(accuracy)
        print('Obervation: %.1f'%((i+1)/10), '| test accuracy: %.4f' % test_accuracy[i])  
        
        
        