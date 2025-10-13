# import torch
# import torch.nn as nn
# import numpy as np
# from math import ceil

# class RelationModule(torch.nn.Module):
#     def __init__(self, clip_model, num_bottleneck, num_frames, dropout_prob=0.5):
#         super(RelationModule, self).__init__()
#         self.num_frames = num_frames
#         self.clip_model = clip_model
#         self.img_feature_dim = clip_model.visual.output_dim
#         self.num_bottleneck = num_bottleneck
#         self.dropout_prob = dropout_prob
#         self.classifier = self.fc_fusion()

#     def fc_fusion(self):
#         return nn.Sequential(
#             nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck * 2),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_prob),
#             nn.Linear(self.num_bottleneck * 2, self.num_bottleneck),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_prob)
#         )

#     def forward(self, input):
#         batch_size, num_frames, _, _, _ = input.size()
#         input = input.view(batch_size * num_frames, 3, 224, 224)

#         with torch.no_grad():  # Freeze CLIP encoder
#             features = self.clip_model.encode_image(input)

#         features = features.float().view(batch_size, num_frames, -1)
#         features = features.view(batch_size, -1)
#         features = self.classifier(features)
#         return features.unsqueeze(1)

# class RelationModuleMultiScale(torch.nn.Module):
#     def __init__(self, clip_model, num_bottleneck, num_frames, rand_relation_sample=False, dropout_prob=0.5):
#         super(RelationModuleMultiScale, self).__init__()
#         self.clip_model = clip_model
#         self.img_feature_dim = clip_model.visual.output_dim
#         self.subsample_num = 9
#         self.scales = [i for i in range(num_frames, 1, -1)]
#         self.dropout_prob = dropout_prob

#         self.relations_scales = []
#         self.subsample_scales = []
#         for scale in self.scales:
#             relations_scale = self.return_relationset(num_frames, scale)
#             self.relations_scales.append(relations_scale)
#             self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

#         self.num_frames = num_frames
#         self.fc_fusion_scales = nn.ModuleList()
#         for scale in self.scales:
#             fc_fusion = nn.Sequential(
#                 nn.Linear(scale * self.img_feature_dim, num_bottleneck * 2),
#                 nn.ReLU(),
#                 nn.Dropout(p=self.dropout_prob),
#                 nn.Linear(num_bottleneck * 2, num_bottleneck),
#                 nn.ReLU(),
#                 nn.Dropout(p=self.dropout_prob)
#             )
#             self.fc_fusion_scales.append(fc_fusion)
            

#         print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])
#         self.rand_relation_sample = rand_relation_sample

#     def forward(self, input):
#         batch_size, num_frames, _, _, _ = input.size()
#         input = input.view(batch_size * num_frames, 3, 224, 224)

#         with torch.no_grad():  # Freeze CLIP encoder
#             features = self.clip_model.encode_image(input)

#         features = features.float().view(batch_size, num_frames, -1)

#         act_all = None
#         act_all_list = []

#         for scaleID, scale in enumerate(self.scales):
#             if scaleID == 0:
#                 act_relation = features[:, self.relations_scales[0][0], :]
#                 act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
#                 act_relation = self.fc_fusion_scales[0](act_relation)
#                 act_all = act_relation.unsqueeze(1)
#                 act_all_list.append(act_relation)
#             else:
#                 act_relation_all = torch.zeros_like(act_all[:, 0])
#                 num_total_relations = len(self.relations_scales[scaleID])
#                 num_select_relations = self.subsample_scales[scaleID]
                
#                 if self.rand_relation_sample:
#                     idx_relations = np.random.choice(num_total_relations, num_select_relations, replace=False)
#                 else:
#                     idx_relations = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

#                 for idx in idx_relations:
#                     act_relation = features[:, self.relations_scales[scaleID][idx], :]
#                     act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
#                     act_relation = self.fc_fusion_scales[scaleID](act_relation)
#                     act_relation_all += act_relation

#                 act_all = torch.cat((act_all, act_relation_all.unsqueeze(1)), 1)
#                 act_all_list.append(act_relation_all)

#         return act_all, act_all_list

#     @staticmethod
#     def return_relationset(num_frames, num_frames_relation):
#         import itertools
#         return list(itertools.combinations(range(num_frames), num_frames_relation))
        
        

# import torch
# import torch.nn as nn
# import numpy as np
# from math import ceil

# class RelationModule(torch.nn.Module):
#     def __init__(self, clip_model, num_bottleneck, num_frames, dropout_prob=0.5):
#         super(RelationModule, self).__init__()
#         self.num_frames = num_frames
#         self.clip_model = clip_model
#         self.img_feature_dim = clip_model.visual.output_dim
#         self.num_bottleneck = num_bottleneck
#         self.dropout_prob = dropout_prob
#         self.classifier = self.fc_fusion()

#     def fc_fusion(self):
#         return nn.Sequential(
#             nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck * 2),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_prob),
#             nn.Linear(self.num_bottleneck * 2, self.num_bottleneck),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_prob)
#         )

#     def forward(self, input):
#         batch_size, num_frames, _, _, _ = input.size()
#         input = input.view(batch_size * num_frames, 3, 224, 224)

#         with torch.no_grad():  # Freeze CLIP encoder
#             features = self.clip_model.encode_image(input)

#         features = features.float().view(batch_size, num_frames, -1)
#         features = features.view(batch_size, -1)
#         features = self.classifier(features)
#         return features.unsqueeze(1)

# class RelationModuleMultiScale(torch.nn.Module):
#     def __init__(self, clip_model, num_bottleneck, num_frames, rand_relation_sample=False, dropout_prob=0.5):
#         super(RelationModuleMultiScale, self).__init__()
#         self.clip_model = clip_model
#         self.img_feature_dim = clip_model.visual.output_dim
#         self.subsample_num = 9
#         self.scales = [i for i in range(num_frames, 1, -1)]
#         self.dropout_prob = dropout_prob

#         self.relations_scales = []
#         self.subsample_scales = []
#         for scale in self.scales:
#             relations_scale = self.return_relationset(num_frames, scale)
#             self.relations_scales.append(relations_scale)
#             self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

#         self.num_frames = num_frames
#         self.fc_fusion_scales = nn.ModuleList()
#         for scale in self.scales:
#             fc_fusion = nn.Sequential(
#                 nn.Linear(scale * self.img_feature_dim, num_bottleneck * 2),
#                 nn.ReLU(),
#                 nn.Dropout(p=self.dropout_prob),
#                 nn.Linear(num_bottleneck * 2, num_bottleneck),
#                 nn.ReLU(),
#                 nn.Dropout(p=self.dropout_prob)
#             )
#             self.fc_fusion_scales.append(fc_fusion)

#         print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])
#         self.rand_relation_sample = rand_relation_sample

#     def forward(self, input):
#         batch_size, num_frames, _, _, _ = input.size()
#         input = input.view(batch_size * num_frames, 3, 224, 224)

#         with torch.no_grad():  # Freeze CLIP encoder
#             features = self.clip_model.encode_image(input)

#         features = features.float().view(batch_size, num_frames, -1)

#         act_all = None
#         act_all_list = []

#         for scaleID, scale in enumerate(self.scales):
#             if scaleID == 0:
#                 act_relation = features[:, self.relations_scales[0][0], :]
#                 act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
#                 act_relation = self.fc_fusion_scales[0](act_relation)
#                 act_all = act_relation.unsqueeze(1)
#                 act_all_list.append(act_relation)
#             else:
#                 act_relation_all = torch.zeros_like(act_all[:, 0])
#                 num_total_relations = len(self.relations_scales[scaleID])
#                 num_select_relations = self.subsample_scales[scaleID]
                
#                 if self.rand_relation_sample:
#                     idx_relations = np.random.choice(num_total_relations, num_select_relations, replace=False)
#                 else:
#                     idx_relations = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

#                 for idx in idx_relations:
#                     act_relation = features[:, self.relations_scales[scaleID][idx], :]
#                     act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
#                     act_relation = self.fc_fusion_scales[scaleID](act_relation)
#                     act_relation_all += act_relation

#                 act_all = torch.cat((act_all, act_relation_all.unsqueeze(1)), 1)
#                 act_all_list.append(act_relation_all)

#         return act_all, act_all_list

#     @staticmethod
#     def return_relationset(num_frames, num_frames_relation):
#         import itertools
#         return list(itertools.combinations(range(num_frames), num_frames_relation))
        
        


import torch
import torch.nn as nn
import numpy as np
from math import ceil

class RelationModuleMultiScale(torch.nn.Module):
    def __init__(self, clip_model, num_bottleneck, num_frames, rand_relation_sample=False, dropout_prob=0.5):
        super(RelationModuleMultiScale, self).__init__()
        self.clip_model = clip_model
        self.img_feature_dim = clip_model.visual.output_dim  # This will be 768 for ViT-L/14
        self.subsample_num = 9
        self.scales = [i for i in range(num_frames, 1, -1)]
        self.dropout_prob = dropout_prob

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList()
        for scale in self.scales:
            fc_fusion = nn.Sequential(
                nn.Linear(scale * self.img_feature_dim, num_bottleneck * 2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_prob),
                nn.Linear(num_bottleneck * 2, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_prob)
            )
            self.fc_fusion_scales.append(fc_fusion)

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])
        self.rand_relation_sample = rand_relation_sample

    def forward(self, input):
        batch_size, num_frames, c, h, w = input.size()
        input = input.view(batch_size * num_frames, c, h, w)

        with torch.no_grad():  # Freeze CLIP encoder
            features = self.clip_model.encode_image(input)

        features = features.float().view(batch_size, num_frames, -1)

        act_all = None
        act_all_list = []

        for scaleID, scale in enumerate(self.scales):
            if scaleID == 0:
                act_relation = features[:, self.relations_scales[0][0], :]
                act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[0](act_relation)
                act_all = act_relation.unsqueeze(1)
                act_all_list.append(act_relation)
            else:
                act_relation_all = torch.zeros_like(act_all[:, 0])
                num_total_relations = len(self.relations_scales[scaleID])
                num_select_relations = self.subsample_scales[scaleID]
                
                if self.rand_relation_sample:
                    idx_relations = np.random.choice(num_total_relations, num_select_relations, replace=False)
                else:
                    idx_relations = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

                for idx in idx_relations:
                    act_relation = features[:, self.relations_scales[scaleID][idx], :]
                    act_relation = act_relation.view(act_relation.size(0), scale * self.img_feature_dim)
                    act_relation = self.fc_fusion_scales[scaleID](act_relation)
                    act_relation_all += act_relation

                act_all = torch.cat((act_all, act_relation_all.unsqueeze(1)), 1)
                act_all_list.append(act_relation_all)

        return act_all, act_all_list

    @staticmethod
    def return_relationset(num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations(range(num_frames), num_frames_relation))