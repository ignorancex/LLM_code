import torch.nn as nn
import torch
# import numpy as np

class PCAProjectNet(nn.Module):
    def __init__(self):
        super(PCAProjectNet, self).__init__()

    def forward(self, features, w_feat, h_feat):     # features: NCWH

        k = features.size(0) * features.size(2)
        x_mean = (features.sum(dim=2).sum(dim=0)/k)
        
        feat_bef = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)
        features = features-x_mean.unsqueeze(0).unsqueeze(2)

        reshaped_features = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)
        
        cls_feat = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous()[:,:,0]

        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        eigval, eigvec = torch.eig(cov, eigenvectors=True)

        cc = 0
        first_compo = eigvec[:,0]
        
        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)[:,:,1:]\
            .view(features.size(0), w_feat, h_feat)

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / torch.abs(maxv + minv)

        cls_relation_list = torch.matmul(first_compo.unsqueeze(0), cls_feat)
        for bs in range(features.size(0)):
            while torch.nonzero((cls_relation_list * (maxv+minv))<0).size(0) > 0:
                cc = cc+1
                first_compo = eigvec[:,cc]
                projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)[:,:,1:]\
                    .view(features.size(0), w_feat, h_feat)
                maxv = projected_map.max()
                minv = projected_map.min()
                projected_map *= (maxv + minv) / torch.abs(maxv + minv)
                cls_relation_list = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)[:,:,0]
        
        return projected_map


if __name__ == '__main__':
    img = torch.randn(6, 512, 14, 14)
    pca = PCAProjectNet()
    pca(img)
