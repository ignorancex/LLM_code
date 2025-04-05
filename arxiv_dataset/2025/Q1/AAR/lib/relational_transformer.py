import numpy as np
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.concept_net_embeddings import load_concept_net
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from einops import reduce

# Abductive Action Transformer
class Relational_Transformer(nn.Module):

    def __init__(self, obj_classes=None, enc_layer_num=None, dec_layer_num=None, semantic=False, concept_net=False, cross_attention=True):

        """
        :param classes: Object classes
        :enc_layer_num: Number of transformer encoder layers
        :dec_layer_num: Number of transformer decoder layers
        :semantic: Whether to use semantic features or not
        :concept_net: Whether to use concept_net embeddings or not
        """
        super(Relational_Transformer, self).__init__()
        self.obj_classes = obj_classes
        self.action_class_num = 157
        self.semantic = semantic
        self.concept_net = concept_net
        self.cross_attention = cross_attention

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        # fully connected layer
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)

        if not concept_net:
            embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)

            self.obj_embed = nn.Embedding(len(obj_classes), 200)
            self.obj_embed.weight.data = embed_vecs.clone()

            self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
            self.obj_embed2.weight.data = embed_vecs.clone()
        else:
            print("Loading concept net embeddings...")
            embed_vecs = load_concept_net(obj_classes, dim=300)

            self.obj_embed = nn.Embedding(len(obj_classes), 300)
            self.obj_embed.weight.data = embed_vecs.clone()

            self.obj_embed2 = nn.Embedding(len(obj_classes), 300)
            self.obj_embed2.weight.data = embed_vecs.clone()

            print("Loaded concept net embeddings...")

        if not self.semantic:
            self.ac_linear = nn.Linear(1536, self.action_class_num)
            self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1536, nhead=8,
                                                  dim_feedforward=2048, dropout=0.1, mode='latter', cross_attention=self.cross_attention)
        else:
            if not self.concept_net:
                if not self.cross_attention:
                    self.ac_linear = nn.Linear(1936, self.action_class_num)
                else:
                    self.ac_linear = nn.Linear(1424, self.action_class_num)

                self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                                          dim_feedforward=2048, dropout=0.1, mode='latter', cross_attention=self.cross_attention)
            else:
                self.ac_linear = nn.Linear(2136, self.action_class_num)
                self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=2136, nhead=8,
                                                      dim_feedforward=2048, dropout=0.1, mode='latter', cross_attention=self.cross_attention)

    def forward(self, entry):

        entry['pred_labels'] = entry['labels']

        # get visual and semantic parts after using a fc-layer
        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)

        # union features
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))

        if not self.cross_attention:
            # true use self attention else cross
            # subject-object representations 
            x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
            
            if self.semantic:
            # semantic part
                subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
                obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]

                subj_emb = self.obj_embed(subj_class) # 200 dim / 300 dim
                obj_emb = self.obj_embed2(obj_class) # 200 dim / 300 dim
                x_semantic = torch.cat((subj_emb, obj_emb), 1)

                rel_features = torch.cat((x_visual, x_semantic), dim=1)

                global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'], pool_type=entry['pool_type'])
            else:
                global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=x_visual, im_idx=entry['im_idx'], pool_type=entry['pool_type'])
        else:
            # using cross-attention
            x_visual = torch.cat((subj_rep, obj_rep), 1)
            
            subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
            obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]

            subj_emb = self.obj_embed(subj_class) # 200 dim / 300 dim
            obj_emb = self.obj_embed2(obj_class) # 200 dim / 300 dim
            x_semantic = torch.cat((subj_emb, obj_emb), 1)

            rel_features = torch.cat((x_visual, x_semantic), dim=1)

            query = vr # union features
            key = rel_features # visual and semantic features of human and object

            features = [query, key]

            global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=features, im_idx=entry['im_idx'], pool_type=entry['pool_type'])

        # fully connected layers  
        entry["relational_feats"] = global_output # max-pooled
        entry["action_class_distribution"] = self.ac_linear(global_output)
        entry["action_class_distribution"] = torch.sigmoid(entry["action_class_distribution"])
        entry["unpooled_relational_feats"] = rel_features
        return entry


class Relational_Transformer_Veri(nn.Module):

    def __init__(self, obj_classes=None, enc_layer_num=None, dec_layer_num=None, semantic=False, concept_net=False):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(Relational_Transformer_Veri, self).__init__()
        self.obj_classes = obj_classes
        self.action_class_num = 157
        self.semantic = semantic
        self.concept_net = concept_net

        # self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        # fully connected layer
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)

        if not concept_net:
            embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)

            self.obj_embed = nn.Embedding(len(obj_classes), 200)
            self.obj_embed.weight.data = embed_vecs.clone()

            self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
            self.obj_embed2.weight.data = embed_vecs.clone()
        else:
            print("Loading concept net embeddings...")
            embed_vecs = load_concept_net(obj_classes, dim=300)

            # print(embed_vecs[1])

            self.obj_embed = nn.Embedding(len(obj_classes), 300)
            self.obj_embed.weight.data = embed_vecs.clone()

            self.obj_embed2 = nn.Embedding(len(obj_classes), 300)
            self.obj_embed2.weight.data = embed_vecs.clone()

            print("Loaded concept net embeddings...")
            # input()
        # self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        # self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        # self.c_rel_compress = nn.Linear(1936, self.contact_class_num)
        if not self.semantic:
            self.ac_linear = nn.Sequential( nn.Linear(1536 + 512, 1536) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(1536, 1))
            #nn.Linear(1536 + 512, 1)
            # remember the embed_dim originally is 1936. But I remove semantic parts therefore - 400 dim to obtain 1536 dim.
            self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1536, nhead=8,
                                                  dim_feedforward=2048, dropout=0.1, mode='latter')
        else:
            if not self.concept_net:
                self.ac_linear = nn.Sequential( nn.Linear(1936 + 512, 1936) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(1936, 1))
                #nn.Linear(1936 + 512, 1)
                self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                                      dim_feedforward=2048, dropout=0.1, mode='latter')
            else:
                # sub -> 300 dim, obj -> 300 dim total 600 + 1536
                self.ac_linear = nn.Sequential( nn.Linear(2136 + 512, 2136) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(2136, 1))
                #nn.Linear(2136 + 512, 1)
                self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=2136, nhead=8,
                                                      dim_feedforward=2048, dropout=0.1, mode='latter')

    def forward(self, entry, class_enc = None):

        # entry = self.object_classifier(entry)
        entry['pred_labels'] = entry['labels']
        # get visual and semantic parts after using a fc-layer
        # visual part
        # print(entry['features'].size())
        # print(entry['pair_idx'].size())
        # print(entry['pair_idx'][:, 0]) # human indices
        # print(entry['pair_idx'][:, 1]) # object indices
        # input()
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))

        # subject-object representations 
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        
        if self.semantic:
        # semantic part
            # print("I'm here!")
            subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
            obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]

            subj_emb = self.obj_embed(subj_class) # 200 dim / 300 dim
            obj_emb = self.obj_embed2(obj_class) # 200 dim / 300 dim
            x_semantic = torch.cat((subj_emb, obj_emb), 1)

            rel_features = torch.cat((x_visual, x_semantic), dim=1)

            global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'], pool_type=entry['pool_type'])
        else:
            global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=x_visual, im_idx=entry['im_idx'], pool_type=entry['pool_type'])

        b = global_output.shape[0]
        score = torch.zeros([b, 157]).to(rel_features.device)        
        for i in range(b):                                        
            score[i] = self.ac_linear(torch.cat([global_output[i].unsqueeze(0).repeat([157,1]),class_enc],dim=1)).squeeze()            
        
        entry["action_class_distribution"] = score
        entry["action_class_distribution"] = torch.sigmoid(entry["action_class_distribution"])

        return entry