from turtle import forward
from more_itertools import tail
import torch
import torch.nn as nn
import math

from lib.word_vectors import obj_edge_vectors
from einops import reduce

#
# Basic Graph Network [Similarity is Jaccard Similarity]
# Obtain representation for each node.
#
class BaseLayer(nn.Module):
    
    def __init__(self, feat_size, emb_size, emb_out):
        super(BaseLayer, self).__init__()
        self.projection = nn.Sequential(nn.Linear(feat_size, emb_size))
        self.graph_feats = nn.Sequential(nn.Linear(emb_size, emb_out), nn.ReLU())

    def forward(self, x ):
        x = self.projection(x)
        A_ = torch.mm(x, x.transpose(dim0=1,dim1=0))
        D = A_.diag().unsqueeze(dim=0)
        Dx = D.repeat([D.shape[1],1])
        Dy = Dx.transpose(dim0=1,dim1=0)
        A = 2 * A_ / (Dx + Dy)        
        graph_weighted_feat = torch.mm(A,x)
        feats = self.graph_feats(graph_weighted_feat)        
        return feats
#
# Same as BaseLayer but query y and value v can be of different size
#
class BaseQVLayer(nn.Module):
    
    def __init__(self, feat_size_x, feat_size_y, emb_size, emb_out_size):
        super(BaseQVLayer, self).__init__()
        self.projection = nn.Sequential(nn.Linear(feat_size_x, emb_size))
        self.projection_y = nn.Sequential(nn.Linear(feat_size_y, emb_size))
        self.graph_feats = nn.Sequential(nn.Linear(emb_size, emb_out_size), nn.ReLU())

    def forward(self, x, y ):
        x = self.projection(x) # value
        y = self.projection_y(y) # query
        A_ = torch.mm(x, y.transpose(dim0=1,dim1=0)) # nv x nq
        Dcol = torch.mm(x, x.transpose(dim0=1,dim1=0)).diag().unsqueeze(dim=1).repeat(1,y.shape[0])
        Drow = torch.mm(y, y.transpose(dim0=1,dim1=0)).diag().unsqueeze(dim=0).repeat(x.shape[0],1)        
        A = 2 * A_ / (Dcol + Drow )        
        graph_weighted_feat = torch.mm(A.transpose(dim0=1,dim1=0),x) # (nq x nv) x (nv x emb_size) --> ()
        feats = self.graph_feats(graph_weighted_feat)        
        return feats

#
# Encoder
#
class EncLayer(nn.Module):
    def __init__(self, feat_size, emb_size, emb_out, dropout = 0.5):
        super(EncLayer, self).__init__()       
        self.baseLayer = BaseLayer(feat_size, emb_size, emb_out)                 

        #self.self_attn = BaseLayer(feat_size, emb_size, emb_out)

        self.linear1 = nn.Linear(feat_size, emb_out)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(emb_out, emb_out)

        self.norm1 = nn.LayerNorm(emb_out)
        self.norm2 = nn.LayerNorm(emb_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        src2 = self.baseLayer(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)        
        return src

#
# Decoder
#
class DecLayer(nn.Module):
    def __init__(self, query_dim, value_dim, emb_size, emb_out, dropout = 0.5):
        super(DecLayer, self).__init__()       
        

        self.multihead2 = BaseQVLayer(value_dim, query_dim, emb_size, emb_out)

        self.linear1 = nn.Linear(emb_out, emb_out)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(emb_out, emb_out)


        self.norm3 = nn.LayerNorm(emb_out)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, query, value):        

        tgt2 = self.multihead2(value, query)
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class GraphED(nn.Module):
    def __init__(self, enc_feat_size, enc_emb_size, enc_emb_out, query_dim, value_dim, dec_emb_size, edc_emb_out):
        super(GraphED, self).__init__()        
        self.encLayer = EncLayer(enc_feat_size, enc_emb_size, enc_emb_out)
        self.decLayer1 = DecLayer(query_dim, value_dim, dec_emb_size, edc_emb_out)
        self.decLayer2 = DecLayer(query_dim, value_dim, dec_emb_size, edc_emb_out)
        self.decLayer3 = DecLayer(query_dim, value_dim, dec_emb_size, edc_emb_out)
    
    def forward(self, x):
        out = self.encLayer(x)
        out =  self.decLayer1(out, out)
        out =  self.decLayer2(out, out)
        out =  self.decLayer3(out, out)        
        return out


class GraphEDWrapper(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, fsize=1936, embed_vecs_flag=True):
        super(GraphEDWrapper, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num
        self.embed_vecs_flag = embed_vecs_flag

        if embed_vecs_flag:
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

            embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        
            self.obj_embed = nn.Embedding(len(obj_classes), 200)
            self.obj_embed.weight.data = embed_vecs.clone()

            self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
            self.obj_embed2.weight.data = embed_vecs.clone()

            self.ac_linear = nn.Linear(fsize, self.action_class_num)

        self.encLayer = EncLayer(fsize, fsize, fsize)
        self.decLayer1 = DecLayer(fsize, fsize, fsize, fsize)
        self.decLayer2 = DecLayer(fsize, fsize, fsize, fsize)
        self.decLayer3 = DecLayer(fsize, fsize, fsize, fsize)

    def forward(self, entry, x=None):

        if x == None:
            entry['pred_labels'] = entry['labels']
            im_idx = entry['im_idx']
            b = int(im_idx[-1] + 1)

            # get visual and semantic parts after using a fc-layer
            subj_rep = entry['features'][entry['pair_idx'][:, 0]]
            subj_rep = self.subj_fc(subj_rep)
            obj_rep = entry['features'][entry['pair_idx'][:, 1]]
            obj_rep = self.obj_fc(obj_rep)

            vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
            vr = self.vr_fc(vr.view(-1,256*7*7))

            # subject-object representations 
            x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

            # semantic representations
            subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
            obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
            subj_emb = self.obj_embed(subj_class) # 200 dim
            obj_emb = self.obj_embed2(obj_class) # 200 dim
            x_semantic = torch.cat((subj_emb, obj_emb), 1)

            rel_features = torch.cat((x_visual, x_semantic), dim=1)

            output = torch.zeros([b, rel_features.shape[1]]).to(rel_features.device)

            for i in range(b):
                # print(rel_features[im_idx == i].size())
                out = self.encLayer(rel_features[im_idx == i])
                out =  self.decLayer1(out, out)
                out =  self.decLayer2(out, out)
                out =  self.decLayer3(out, out)

                if entry['pool_type'] == 'max':
                    output[i, :] = reduce(out, 'n d -> d', 'max')
                elif entry['pool_type'] == 'avg':
                    output[i, :] = reduce(out, 'n d -> d', 'mean')
                else:
                    pass

            entry["relational_feats"] = output
            entry["action_class_distribution"] = torch.sigmoid(self.ac_linear(output))
            entry["unpooled_relational_feats"] = rel_features

            return entry

        else:
            if self.embed_vecs_flag:
                out =  self.encLayer(x)
                out =  self.decLayer1(out, out)
                out =  self.decLayer2(out, out)
                out =  self.decLayer3(out, out)
            else:
                out = self.encLayer(x)
                out = self.decLayer1(out, out)
            return out


#
# Head is the subject representation and tail is the object representation
# out size is half of the head_size + tailsize
class RBP(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, head_size=712, tail_size=712, out_size=712):
        super(RBP, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num

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

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.bimodel = nn.Bilinear(head_size, tail_size, out_size)
        self.linear = nn.Linear(head_size + tail_size, out_size)
        self.bimodel_projection = nn.Linear(2 *out_size , 2 *out_size)
        self.relu = torch.nn.ReLU()
        
        self.ac_linear = nn.Linear(1936, self.action_class_num)

    def forward(self, entry):

        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)
        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in a single frame

        # get visual and semantic parts after using a fc-layer
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))

        # semantic representations
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class) # 200 dim
        obj_emb = self.obj_embed2(obj_class) # 200 dim

        # subject representations  + semantics
        subj_comb = torch.cat((subj_rep, subj_emb), 1)

        # # object representations + semantics
        obj_comb = torch.cat((obj_rep, obj_emb), 1)

        output = torch.zeros([b, 1936]).to(obj_comb.device)

        # for transformer sequence prediction
        memory = torch.zeros([l, b, 1936]).to(obj_comb.device)
        memory_key_padding_mask = torch.zeros([b, l], dtype=torch.uint8).to(obj_comb.device)

        # head is the subj rep, tail is the obj rep
        for i in range(b):
            indiv_s_rep = subj_comb[im_idx == i]
            indiv_o_rep = obj_comb[im_idx == i]

            out = self.bimodel(indiv_s_rep, indiv_o_rep)
            joint = torch.cat([indiv_s_rep, indiv_o_rep],dim=1)
            z = self.linear(joint)
            z = self.relu(torch.cat([z,out],dim=1))
            z = self.bimodel_projection(z)

            rel_features = torch.cat((z, vr[im_idx == i]), 1)

            memory[:rel_features.size(0), i, :] = rel_features
            memory_key_padding_mask[i, rel_features.size(0):] = 1

            if entry['pool_type'] == 'max':
                output[i, :] = reduce(rel_features, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                output[i, :] = reduce(rel_features, 'n d -> d', 'mean')

        entry["relational_feats"] = output # max pooled features
        entry["action_class_distribution"] = torch.sigmoid(self.ac_linear(output))
        entry["unpooled_relational_feats"] = [memory, memory_key_padding_mask]
        return entry

# This one has 3 GraphEDWrapper for head, tail, and joint.
# All size depends in out
# Joint model has head out_size output to reduce the parameters.
# Only use object semantics
# We use 512 out_size
class BiGED(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, head_size=712, tail_size=712, out_size=512):
        super(BiGED, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num
        # self.graph_flag = graph_flag

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
        self.subj_fc = nn.Linear(2048, out_size)
        self.obj_fc = nn.Linear(2048, out_size)
        self.vr_fc = nn.Linear(256*7*7, out_size)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()        
        
        self.tail_projection = GraphEDWrapper(obj_classes, fsize= (out_size + 200), embed_vecs_flag=False)
        

        self.bimodel = nn.Bilinear( out_size  , (out_size + 200), out_size)
        
        # Join projection model.
        self.joint_linear = nn.Linear(out_size + out_size + 200, out_size)
        self.join_projection = GraphEDWrapper(obj_classes, fsize= out_size , embed_vecs_flag=False)       
        
        # Final layer
        self.classifier = nn.Sequential(nn.Linear(out_size + out_size + out_size, out_size + out_size + out_size) , 
                                        torch.nn.ReLU() , 
                                        nn.Linear(out_size + out_size + out_size, self.action_class_num))
        
    def forward(self, entry):
        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)
        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in a single frame

        # get visual and semantic parts after using a fc-layer
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep) # out_size
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep) # out_size

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7)) # out_size

        # semantic representations
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]        
        obj_emb = self.obj_embed(obj_class) # 200 dim

        # subject representations  + semantics
        # do not include semantics
        subj_comb = subj_rep #torch.cat((subj_rep, subj_emb), 1) # out_size

        # # object representations + semantics
        obj_comb = torch.cat((obj_rep, obj_emb), 1) # out_size + 200
        output = torch.zeros([b, 157]).to(obj_comb.device)
        all_feats = torch.zeros([b, 1536]).to(obj_comb.device)

        # for transformer sequence prediction
        memory = torch.zeros([l, b, 1536]).to(obj_comb.device)
        memory_key_padding_mask = torch.zeros([b, l], dtype=torch.uint8).to(obj_comb.device)

        for i in range(b):
            indiv_s_rep = subj_comb[im_idx == i] # out_size
            indiv_o_rep = obj_comb[im_idx == i] # out_size + 200

            # head is subj, tail is obj
            head_x = indiv_s_rep     #    out_size - only subj visual rep
            tail_x = self.tail_projection(None, indiv_o_rep)  # out_size + 200  - obj visual and semantic
            # bilinear takes in subj visual rep and obj comb rep
            bi_out = self.bimodel(head_x,tail_x)        

            joint = self.joint_linear(torch.cat([head_x,indiv_o_rep],dim=1)) # size = out_size
            joint_out = self.join_projection(None, joint)     # size = out_size                 

            rel_features = torch.cat([joint_out, bi_out, vr[im_idx == i]],dim=1)   #    # size = 3 * out_size     

            memory[:rel_features.size(0), i, :] = rel_features
            memory_key_padding_mask[i, rel_features.size(0):] = 1

            if entry['pool_type'] == 'max':
                all_feats[i, :] = reduce(rel_features, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                all_feats[i, :] = reduce(rel_features, 'n d -> d', 'mean')

            rel_features = self.classifier(rel_features) # this will do classification as well.

            if entry['pool_type'] == 'max':
                output[i, :] = reduce(rel_features, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                output[i, :] = reduce(rel_features, 'n d -> d', 'mean')
                
        entry["relational_feats"] = all_feats 
        entry["action_class_distribution"] = torch.sigmoid(output)
        entry["unpooled_relational_feats"] = [memory, memory_key_padding_mask]

        return entry

class GNNED_Veri(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, fsize=1936, embed_vecs_flag=True):
        super(GNNED_Veri, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num

        if embed_vecs_flag:
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

            embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        
            self.obj_embed = nn.Embedding(len(obj_classes), 200)
            self.obj_embed.weight.data = embed_vecs.clone()

            #self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
            #self.obj_embed2.weight.data = embed_vecs.clone()

            self.ac_linear = nn.Sequential( nn.Linear(fsize + 512, fsize + 512) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(fsize + 512, 1))
            #self.ac_linear = nn.Linear(fsize + 512, 1)

        self.encLayer = EncLayer(fsize, fsize, fsize)
        self.decLayer1 = DecLayer(fsize, fsize, fsize, fsize)
        self.decLayer2 = DecLayer(fsize, fsize, fsize, fsize)
        self.decLayer3 = DecLayer(fsize, fsize, fsize, fsize)

    def forward(self, entry, class_enc = None):        
        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)

        # get visual and semantic parts after using a fc-layer
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))

        # subject-object representations 
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic representations
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class) # 200 dim
        obj_emb = self.obj_embed(obj_class) # 200 dim
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)

        #output = torch.zeros([b, rel_features.shape[1]]).to(rel_features.device)
        score = torch.zeros([b, 157]).to(rel_features.device)

        for i in range(b):            
            out = self.encLayer(rel_features[im_idx == i])
            out =  self.decLayer1(out, out)
            out =  self.decLayer2(out, out)
            out =  self.decLayer3(out, out)

            if entry['pool_type'] == 'max':
                output_ = reduce(out, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                output_ = reduce(out, 'n d -> d', 'mean')
            else:
                pass
            
            infeature = torch.cat([output_.unsqueeze(0).repeat([157,1]),class_enc],dim=1)               
            score[i] = self.ac_linear(infeature).squeeze()

        entry["action_class_distribution"] = torch.sigmoid(score)

        return entry


class RBP_Veri(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, head_size=712, tail_size=712, out_size=712):
        super(RBP_Veri, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num

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

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.bimodel = nn.Bilinear(head_size, tail_size, out_size)
        self.linear = nn.Linear(head_size + tail_size, out_size)
        self.bimodel_projection = nn.Linear(2 *out_size , 2 *out_size)
        self.relu = torch.nn.ReLU()
        
        self.ac_linear = nn.Sequential( nn.Linear(1936 + 512, 1936) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(1936, 1))
        #nn.Linear(1936 + 512, 1)

    def forward(self, entry, class_enc = None):

        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)

        # get visual and semantic parts after using a fc-layer
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))

        # semantic representations
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class) # 200 dim
        obj_emb = self.obj_embed2(obj_class) # 200 dim

        # subject representations  + semantics
        subj_comb = torch.cat((subj_rep, subj_emb), 1)

        # # object representations + semantics
        obj_comb = torch.cat((obj_rep, obj_emb), 1)

        output = torch.zeros([b, 157]).to(obj_comb.device)

        # head is the subj rep, tail is the obj rep
        for i in range(b):
            indiv_s_rep = subj_comb[im_idx == i]
            indiv_o_rep = obj_comb[im_idx == i]

            out = self.bimodel(indiv_s_rep, indiv_o_rep)
            joint = torch.cat([indiv_s_rep, indiv_o_rep],dim=1)
            z = self.linear(joint)
            z = self.relu(torch.cat([z,out],dim=1))
            z = self.bimodel_projection(z)

            rel_features = torch.cat((z, vr[im_idx == i]), 1)

            if entry['pool_type'] == 'max':
                output_ = reduce(rel_features, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                output_ = reduce(rel_features, 'n d -> d', 'mean')
            else:
                pass
            infeature = torch.cat([output_.unsqueeze(0).repeat([157,1]),class_enc],dim=1)           
            output[i] = torch.sigmoid(self.ac_linear(infeature).squeeze())

        entry["action_class_distribution"] = output

        return entry    


class BiGED_Veri(nn.Module):
    def __init__(self, obj_classes, action_class_num=157, head_size=712, tail_size=712, out_size=256):
        super(BiGED_Veri, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num
        # self.graph_flag = graph_flag

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
        self.subj_fc = nn.Linear(2048, out_size)
        self.obj_fc = nn.Linear(2048, out_size)
        self.vr_fc = nn.Linear(256*7*7, out_size)

        # TODO: set the embeddings to false and see what we get
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()        
        

        
        self.tail_projection = GraphEDWrapper(obj_classes, fsize= (out_size + 200), embed_vecs_flag=False)
        

        self.bimodel = nn.Bilinear( out_size  , (out_size + 200), out_size)
        
        # Join projection model.
        self.joint_linear = nn.Linear(out_size + out_size + 200, out_size)
        self.join_projection = GraphEDWrapper(obj_classes, fsize= out_size , embed_vecs_flag=False)       
        
        # Final layer
        self.classifier = nn.Sequential( nn.Linear(out_size + out_size + out_size + 512, out_size + out_size + out_size) , 
                                                torch.nn.ReLU() , 
                                                nn.Linear(out_size + out_size + out_size, 1))

    def forward(self, entry, class_enc = None):
        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)

        # get visual and semantic parts after using a fc-layer
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep) # out_size
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep) # out_size

        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7)) # out_size

        # semantic representations
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]        
        obj_emb = self.obj_embed(obj_class) # 200 dim

        # subject representations  + semantics
        subj_comb = subj_rep #torch.cat((subj_rep, subj_emb), 1) # out_size

        # # object representations + semantics
        obj_comb = torch.cat((obj_rep, obj_emb), 1) # out_size + 200
        output = torch.zeros([b, 157]).to(obj_comb.device)

        
        for i in range(b):
            indiv_s_rep = subj_comb[im_idx == i] # out_size
            indiv_o_rep = obj_comb[im_idx == i] # out_size + 200

            # head is subj, tail is obj
            head_x = indiv_s_rep     #    out_size 
            tail_x = self.tail_projection(None, indiv_o_rep)     # out_size + 200  
            bi_out = self.bimodel(head_x,tail_x)        

            joint = self.joint_linear(torch.cat([head_x,indiv_o_rep],dim=1)) # size = out_size
            joint_out = self.join_projection(None, joint)     # size = out_size                 

            rel_features = torch.cat([joint_out, bi_out, vr[im_idx == i]],dim=1)   #    # size = 3 * out_size                 

            if entry['pool_type'] == 'max':
                rel_features = reduce(rel_features, 'n d -> d', 'max')
            elif entry['pool_type'] == 'avg':
                rel_features = reduce(rel_features, 'n d -> d', 'mean')
            else:
                pass   
            infeature = torch.cat([rel_features.unsqueeze(0).repeat([157,1]),class_enc],dim=1)           
            output[i] = self.classifier(infeature).squeeze()
        entry["action_class_distribution"] = torch.sigmoid(output)

        return entry


# Rule-based Inference
class Relational(nn.Module):
    def __init__(self, obj_classes, action_class_num=157):
        super(Relational, self).__init__()

        self.obj_classes = obj_classes
        self.action_class_num = action_class_num 
        self.map = {}

    def get_object_string(self, objects):
        A = []
        for x in objects:
            A.append(x.item())
        A.sort()        
        s = ''
        for a in range(len(A)):
            if a == 0:
                s = str(A[a])
            else:
                s =  s + '-' +str(A[a])        
        return s


    def forward(self, entry):
        entry['pred_labels'] = entry['labels']
        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)            
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]  
        if self.training:
            actions = entry['gt_action_class_label']                           
        output = torch.zeros([b, 157])
        for i in range(b):
            objects = obj_class[im_idx == i] 
            object_str = self.get_object_string(objects) 
            if self.training:         
                if object_str in self.map.keys():
                    self.map[object_str] = self.map[object_str] + actions[i].float()
                else:
                    self.map[object_str] = actions[i].float()                
            else:
               if object_str in self.map.keys(): 
                    output[i] = self.map[object_str]
                    output[i] = output[i] / output[i].sum()
        entry["action_class_distribution"] = torch.sigmoid(output)
        return entry