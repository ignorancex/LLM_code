"""
Let's get the relationships yo
"""

from typing import Dict, List, Any

import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
# from models.multiatt.net_utils import FC, MLP, LayerNorm
import torch, math
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
import os
# import pickle
import pickle
import ipdb
#######################################3
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# from models.multiatt.mca import SA, SGA,  MCA_ED, MCA_SK, AttFlat

#######################################
SAVE_ROOT = "/phys/ssd/tangxueq/tmp/vcr/vcrimage/rationale"

@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,

                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 # lstm_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int =1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQA, self).__init__(vocab)
################################################################################################################################
        self.dic_size = 200

        path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell'
        m1 = os.path.join(path, 'memory_cell1' + '.npz')
        m2 = os.path.join(path, 'memory_cell2' + '.npz')
        self.memory_cell1, self.memory_cell2 = self.memory_cell_load(m1, m2)
        self.memory_cell1 = torch.nn.Parameter(torch.Tensor(self.memory_cell1))
        self.memory_cell2 = torch.nn.Parameter(torch.Tensor(self.memory_cell2))

        torch.save(self.memory_cell1, m1)
        print(self.memory_cell1)

        # self.memory_cell2 = torch.nn.Parameter(torch.Tensor(self.memory_cell_load(m1,m2)))
        torch.save(self.memory_cell2, m2)
        print(self.memory_cell2)

        ###############################################################################################################################
        # if os.path.isfile(self.memory_cell_path):
        #     print('load memory_cell from {0}'.format(self.memory_cell_path))
        #     memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
        # else:
        #     print('create a new memory_cell')
        #     memory_init = np.random.rand(1000, 1024) / 100
        # memory_init = np.float32(memory_init)
        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()


#################################################################################################################################

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################
        INPUT_SIZE =512 #768
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=1024,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
        )





        self.lstm_norm = torch.nn.BatchNorm1d(200)
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.span_reshape = TimeDistributed(torch.nn.Linear(512, 768))
        self.qao_reshape = torch.nn.Linear(512,768)

        # self.out = nn.Linear(50, 1)
        # self.reasoning_encoder = TimeDistributed(lstm_encoder)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question

        #[96,4,50,1024]
        dim = 64#1024#768*2 #sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        # (span_encoder.get_output_dim(), self.pool_answer),
                                        # (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            # torch.nn.Dropout(input_dropout, inplace=False),
            # torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, 1),
        )

        self.bceLoss = nn.BCEWithLogitsLoss()




        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(dim, hidden_dim_maxpool),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(hidden_dim_maxpool, 4),
        # )
        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(dim,1))
        self.linear_l = torch.nn.Linear(512,64)
        self.linear_i = torch.nn.Linear(512, 256)


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
      #####################################################################
        self.stacking = MCA_SK()
        self.ED = MCA_ED()
        self.END = MCA_END()

        self.aq_AttFlat = AttFlat(512)
        self.qa_AttFlat = AttFlat(512)
        self.GA_transformer = GA_transformer()
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
        self.encoder_layer_vc = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder_vc = nn.TransformerEncoder(self.encoder_layer_vc, num_layers=3)
        self.proj_norm = LayerNorm(64)
        self.proj = nn.Linear(128, 4)

        self.clstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            # bidirectional=True

        )

        self.ilstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            # bidirectional=True

        )


        initializer(self)

        ##############################################################################
        # self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')

    def memory_cell_load(self, m1, m2):
        if os.path.isfile(m1) and os.path.isfile(m2):
            print('load memory_cell from {0}'.format(m1), '{}'.format(m2))

            # memory_init = np.load(self.memory_cell_path)['memory_cell']
            memory_init1 = torch.load(m1)  # ['memory_cell1']#[()]

            memory_init2 = torch.load(m2)  # ['memory_cell2']#[()]
        else:
            print('create a new memory_cell')
            # memory_init = np.random.rand(10000, 1024)/ 100
            # memory_init = torch.random(10000,1024)/100
            memory_init1 = torch.randn(self.dic_size, 64) / 100  # randn
            memory_init2 = torch.randn(self.dic_size, 64) / 100
        memory_init1 = memory_init1.float()
        memory_init2 = memory_init2.float()
        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()

        return memory_init1, memory_init2


    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        # x = self.span_encoder(span_rep, span_mask)
        # x = self.span_reshape(x)

        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def padding(self,q_rep,a_rep):
        # max_len = max(max([i.size(2) for i in q_rep]), max([i.size(2) for i in a_rep]))
        max_len = max(a_rep.size(2),q_rep.size(2))
        a1, b1, c1, d1 = a_rep.size()
        a2, b2, c2, d2 = q_rep.size()
        padding_a = torch.zeros(a1, b1, max_len - c1, d1).float().cuda()
        padding_q = torch.zeros(a2, b2, max_len - c2, d2).float().cuda()


        q_rep_new = torch.cat((q_rep, padding_q), dim=2)
        a_rep_new = torch.cat((a_rep, padding_a), dim=2)

        qa_rep = torch.cat((q_rep_new, a_rep_new), dim=3)  # [batch_size, 8, seq_len, 1536]

        return qa_rep

    def Dictionary(self, h, M):  # [96*4,768] M[10000,768]

        att = h @ M.T
        att = self.lstm_norm(att)

        att = F.softmax(att, dim=1)  # [96*4,10000]
        #############################################################3


        ################################################################
        # print('att1:', att.backward())
        # print('att1:', att.backward())
        # print("att_res :",att.grad)
        # output_res = torch.mm(att, self.memory_cell)  # [96*4,10000]*[10000,768]    ->   #[96*4,768]
        Toutput_res = att @ M

        return Toutput_res

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self,


                # images: torch.Tensor,
                obj_reps: Dict[str, torch.Tensor],
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        # print("question:\n", len(question), "\n")
        # print(question['bert'].size())
        # print("answers:\n", len(answers), "\n")
        # print(answers['bert'].size())


        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]



        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))



        #---------------------------------------------------------------------------------------

        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        # print(q_rep.size(), 'q_rep.size()')
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        # Perform Q by A attention
        # [batch_size, answer number, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)  # formula
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                                        a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:, None, None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))
        # attended_q = self.linear_l(attended_q)
        # attended_o = self.linear_l(attended_o)

        qa_rep = self.GA_transformer(attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),None,None) #[b x 4, q_len, 512]
        # qa2_rep = self.GA_transformer(attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),None,None) #[b x 4, q_len, 512]
        # qa3_rep = self.GA_transformer(attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),None,None) #[b x 4, q_len, 512]
        aq_rep = self.GA_transformer(attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),None,None) #[b x 4, q_len, 512]
        # qa_rep,aq_rep = self.ED(attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),None,None)
        # aq_rep = self.SA(attended_q.view(attended_q.size(0)*attended_q.size(1),attended_q.size(2),attended_q.size(3)),attended_o.view(attended_o.size(0)*attended_o.size(1),attended_o.size(2),attended_o.size(3)),None,None)
        #-------------------------------

        qa_rep = torch.cat((qa_rep,aq_rep),dim=1)
        qa_rep = torch.transpose(qa_rep,1,2).contiguous()

        output = self.AvgPool(qa_rep)

        output = torch.transpose(output, 1, 2).contiguous()
        output_res = output.squeeze()  # [96 x 4, 512]
        output_res = self.linear_l(output_res)




        # qa_rep = torch.cat((qa1_rep,qa2_rep,qa3_rep),dim=1)

        # qa_rep = self.transformer_encoder_vc(qa_rep)



        # qa_rep_feature = self.qa_AttFlat(qa_rep,None)
        # aq_rep_feature = self.aq_AttFlat(aq_rep,None)
        # feature = qa_rep_feature + aq_rep_feature
        # output_res = self.proj_norm(feature)

        output_res_d = self.Dictionary(output_res,self.memory_cell1)

        output_res = output_res_d.view(-1, 4, 64)




#--------------------------------------------------------------------------------------------------------------------
        # qa_rep = self.transformer_encoder_vc(qa_rep)
        # #-----------------------------------
        # qa_rep_res = torch.transpose(qa_rep,1,2).contiguous()
        #
        # qa_rep_res = self.AvgPool(qa_rep_res)  #[b x 4, 512, 1]
        # qa_rep_res = qa_rep_res.squeeze(-1)
        # qa_feature = self.linear_i(qa_rep_res) #[b x 4, 256]
        # out = self.Dictionary(qa_feature,self.memory_cell1)
        #
        # output_res = out.view(-1,4,256)



        #---------------------------------------------------------------------------------------------


#
#
#         # obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
#
# ##################################################################################################################
#         '''try idea to create new model with dictionary'''
#
#         '''padding q_rep and a_rep with 0'''
#         # max_len = max(max([i.size(2) for i in question['bert']]),max([i.size(2) for i in answers['bert']]))
#         # a, b, c, d = a_rep.size()
#         # padding = torch.zeros(a,b,max_len,d).float()
#         # q_rep_new = torch.cat((q_rep,padding),dim = 2)
#         # a_rep_new = torch.cat((a_rep,padding), dim = 2)
#         # print(question['bert'].size())
#         q_feature = question['bert']  # [b, 4, len, 768]   #[b, num, 512]
#         a_feature = answers['bert']  # [b, 4, len, 768]
#
#         # qa_feature =  torch.cat((question['bert'],answers['bert']),dim = 2)
#
#         q_feature = q_feature.view(q_feature.size(0)*q_feature.size(1),-1,q_feature.size(-1)) #[b*4, len, 768]
#         a_feature = a_feature.view(a_feature.size(0)*a_feature.size(1), -1, a_feature.size(-1)) #[b*4, len, 768]
#
#         # qa_feature = qa_feature.view(qa_feature.size(0),-1,qa_feature.size(-1))
#
#         image_feature = obj_reps['obj_reps']    #[b, num, 512]
#
#         # image_feature = torch.unsqueeze(image_feature,1)
#         # image_feature = image_feature.repeat(1,4,1,1)
#
#         # image_feature = image_feature.view(image_feature.size(0)*image_feature.size(1),image_feature.size(2),image_feature.size(3))
#         # print(a_feature.size(), 'jjjjjjjjjjjjjjjjjj')
#
#
#
#         image_mask = self.make_mask(image_feature)
#         # print(qa_mask.shape,image_mask.shape)
#
#         # q_feature = self.linear_l(q_feature)
#         # a_feature = self.linear_l(a_feature)
#         # image_feature = self.linear_i(image_feature)
#         q_feature_rep, q_image_feature_rep = self.stacking( q_feature, image_feature, None, None)#[b*4, len, 768]   #[b*4, num, 512]
#         a_feature_rep, a_image_feature_rep = self.stacking( a_feature, image_feature, None, None)
# #///////////////////////////////////////////////////////////////////////////////////////////////////
#
#
#
#
#
# #//////////////////////////////////////////////////////////////////////////////////////////////////
#
# #--------------------------------------------------------------------------------------------------------------------
# #------------------------------ for loop----------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------------------
#
#         # q_feature_list = []
#         # a_feature_list = []
#         # q_image_list = []
#         # a_image_list = []
#         #
#         # for i in range(4):
#         #
#         #     q_feature_i = q_feature[:, i, :, :]
#         #     q_feature_i = q_feature_i.squeeze()
#         #     a_feature_i = a_feature[:, i, :, :]
#         #     a_feature_i = a_feature_i.squeeze()
#         #     q_mask = self.make_mask(q_feature_i)  # torch.cat((question_mask,answer_mask),dim = -1)
#         #     a_mask = self.make_mask(a_feature_i)  # torch.cat((question_mask,answer_mask),dim = -1)
#         #
#         #     q_feature_rep_i, q_image_feature_rep_i = self.ED(q_feature_i,image_feature,None,None)   #[b, 4xlen, 768]   [b, num, 512]
#         #     q_feature_rep_i = torch.unsqueeze(q_feature_rep_i, 1)
#         #     q_image_feature_rep_i = torch.unsqueeze(q_image_feature_rep_i, 1)
#         #
#         #
#         #     q_feature_list.append(q_feature_rep_i)
#         #     q_image_list.append(q_image_feature_rep_i)
#         #
#         #     a_feature_rep_i, a_image_feature_rep_i = self.ED(a_feature_i,image_feature,None,None)
#         #     a_feature_rep_i = torch.unsqueeze(a_feature_rep_i, 1)
#         #     a_image_feature_rep_i = torch.unsqueeze(a_image_feature_rep_i, 1)
#         #
#         #     #
#         #     a_feature_list.append(a_feature_rep_i)
#         #     a_image_list.append(a_image_feature_rep_i)
#         #
#         #
#         # q_feature_rep = torch.cat((q_feature_list),dim = 1)
#         # a_feature_rep = torch.cat((a_feature_list),dim = 1)
#         # q_image_feature_rep = torch.cat((q_image_list),dim=1)
#         # a_image_feature_rep = torch.cat((a_image_list),dim=1)
#         #
#         #
#
#
# ########### Attention reduce --------------------------------------------------------
#         # q_feature_rep = q_feature_rep.view(q_feature_rep.size(0),4,-1,768)   #[b, 4, len, 768]
#         # a_feature_rep = a_feature_rep.view(a_feature_rep.size(0),4,-1,768)
#
#
#         qa_feature_rep = torch.cat((q_feature_rep,a_feature_rep),dim = 1)#[bx4, len, 768]
#
#
#         # qa_feature_rep = qa_feature_rep.view(qa_feature_rep.size(0)*qa_feature_rep.size(1),-1,768)  #[bx4, len, 768]
#         qa_image_rep = torch.cat((q_image_feature_rep,a_image_feature_rep),dim=1)                   #[b, num, 521]
#         # qa_image_rep = qa_image_rep.view(qa_image_rep.size(0) * qa_image_rep.size(1), -1, 512)
#
#
#         # qa_image_mask = self.make_mask(qa_image_rep)
#         # qa_feature_mask = self.make_mask(qa_feature_rep)
#
# # # ##############################################################################################################################
# #
# #         qa_clstm,(h1,c1) = self.clstm(qa_feature_rep)        #
# #         qa_clstm = qa_clstm[:,-1,:]        #
# #
# #         qa_ilstm,(h2,c2) = self.ilstm(qa_image_rep)
# #         qa_ilstm = qa_ilstm[:,-1,:]
# #         # qa_ilstm = qa_ilstm.view(-1,4,512)#
# #
# #             #
# #
# #       #######################################################################
# #       #---------dict--------------------------------------------------------
# #         qa_clstm = self.Dictionary(qa_clstm,self.memory_cell1)
# #         qa_ilstm = self.Dictionary(qa_ilstm,self.memory_cell2)
# #
# #
# #       ####################################################################
# #         qa_clstm = qa_clstm.view(-1, 4, 128)
# #
# #         qa_ilstm = torch.unsqueeze(qa_ilstm, 1)
# #         qa_ilstm = qa_ilstm.repeat(1, 4, 1)
# #         output_res = torch.cat((qa_clstm,qa_ilstm),dim = -1)
# #         # output_res = qa_clstm
# ####################################################################################################
#         #
#         qa_image_res = self.image_AttFlat(qa_image_rep,None)   #[b,dim]
#         qa_feature_res = self.qa_AttFlat(qa_feature_rep,None)
#
#         qa_feature_res = qa_feature_res.view(-1,4,64)
#         # qa_image_res = qa_image_res.view(-1,4,128)
#         qa_image_res = torch.unsqueeze(qa_image_res,1)
#         qa_image_res = qa_image_res.repeat(1,4,1)
#
#         qa_proj_feat = qa_feature_res + qa_image_res
#
#
#         proj_feat = self.proj_norm(qa_proj_feat)
#         # proj_feat = torch.sigmoid(self.proj(proj_feat))
#
#
#         output_res = proj_feat#self.proj(proj_feat_feat)

###################################################################################################
        logits = self.final_mlp(output_res).squeeze(-1)#self.final_mlp(output_res)
        # print(logits.shape, 'jjjjjjjjjjjj')
        # logits = output_res#.squeeze(2       [94,4]
        # print(logits)
        class_probabilities = F.softmax(logits,dim = -1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }


        if label is not None:

            loss = self._loss(logits, label.long().view(-1))

        ############################################################################

        # binary_label = torch.zeros(label.size(0), 4)
        # if label is not None:
        #     # label[batch]    label.long().view(-1)[batch]   logits[batch, num_answes]
        #
        #     for i in range(label.size(0)):
        #
        #         label_per_batch = torch.zeros(4)
        #         if label[i] == 0:
        #             label_per_batch[0] = 1
        #         elif label[i] == 1:
        #             label_per_batch[1] = 1
        #         elif label[i] == 2:
        #             label_per_batch[2] = 1
        #         elif label[i] == 3:
        #             label_per_batch[3] = 1
        #         # print("ggg",binary_label[i])
        #
        #         binary_label[i, :] = label_per_batch




            # loss = self.bceLoss(logits, binary_label.cuda())




        ##############################################################################







            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]



        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}




class MHAtt(nn.Module):
    def __init__(self,dim1, dim2):
        super(MHAtt, self).__init__()

        self.hidden_size = dim1#  512
        # self.dropout = 0.1
        # self.multi_head = 8
        self.hidden_size_head = int(dim1 / 8)
        self.linear_v = nn.Linear(dim2, dim1 )
        self.linear_k = nn.Linear(dim2, dim1)
        self.linear_q = nn.Linear( dim1,dim1)
        self.linear_merge = nn.Linear(dim1,dim1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, v, k, q, mask):

        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            8 ,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            8,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            8,
            self.hidden_size_head
        ).transpose(1, 2)


        atted = self.att(v, k, q, mask)


        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
#----------------------------------------------------------------------
#-----------------------------这里mask处理需要再想想_--------------------
#----------------------------------------------------------------------

            scores = scores.masked_fill(mask, -1e9)

        # print(scores.shape,'kkkkkkkkkkkkk')
        att_map = F.softmax(scores, dim=-1)

        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        # self.hidden_size = hidden_size

        self.mlp = MLP(
            in_size= hidden_size,
            mid_size= 64,#hidden_size*1,
            out_size= hidden_size,
            dropout_r= 0.1,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self,dim):
        super(SA, self).__init__()

        # self.hidden_size = dim

        self.mhatt = MHAtt(dim,dim)
        self.ffn = FFN(dim)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(dim)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(dim)

    def forward(self, x, x_mask):

        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))


        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self,dim1,dim2):
        super(SGA, self).__init__()
        # self.hidden_size = dim1

        self.mhatt1 = MHAtt(dim1,dim1)
        self.mhatt2 = MHAtt(dim1,dim2)
        self.ffn = FFN(dim1)


        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(dim1)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(dim1)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(dim1)

    def forward(self, x, y, x_mask, y_mask):
        ''' x: image， y: context, '''


        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self):
        super(MCA_ED, self).__init__()
        # self.layer = 6
        # self.enc_list = nn.ModuleList([SA(768) for _ in range(3)])
        # self.dec_list = nn.ModuleList([SGA(512,768) for _ in range(3)])

        self.enc_list = nn.ModuleList([SA(512) for _ in range(3)])
        self.dec_list = nn.ModuleList([SGA(512, 512) for _ in range(3)])


    def forward(self, x, y, x_mask, y_mask):
        # X： text, Y: image
        for enc in self.enc_list:
            x = enc(x, x_mask)

        # x = x.view(-1,4*x.size(1),x.size(-1))
        # x_mask_res = self.make_mask(x)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y


class MCA_END(nn.Module):
    def __init__(self):
        super(MCA_END, self).__init__()
        # self.layer = 6
        self.enc_list = nn.ModuleList([SGA(768,512) for _ in range(1)])
        self.dec_list = nn.ModuleList([SGA(512,768) for _ in range(1)])

    def forward(self, x, y, x_mask, y_mask):
        # X： 文字, Y: image
        for enc in self.enc_list:
            x_res = enc(x, y, x_mask, y_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)


        return x_res, y
#-----------------------------------------------------------------
#-----------------GA_Transformer ---------------------------------
#-----------------------------------------------------------------

class GA_transformer(nn.Module):
    def __init__(self):
        super(GA_transformer,self).__init__()
        self.enc_list = nn.ModuleList([SGA(512,512) for _ in range(4)])

    def forward(self, x, y, x_mask, y_mask):
        # X： a, Y: q
        for enc in self.enc_list:
            y = enc(y, x, y_mask, x_mask)
        return y

# ------------------------------------------------
# ---- MAC Layers Cascaded by Stacking ----
# -----------------------------------------------

class MCA_SK(nn.Module):
    def __init__(self):
        super(MCA_SK, self).__init__()
        self.num_unit = 3

        # self.SA = SA(768)
        # self.SGA = SGA(512,768)

        self.SA = SA(512)
        self.SGA = SGA(512, 512)


    def MCA_layer(self, x, y, x_mask, y_mask):
        # X： 文字, Y: image

        x = self.SA(x, x_mask)

        y = self.SGA(y, x, y_mask, x_mask)#   map contex to image dimention

        return x, y

    def forward(self, x, y, x_mask, y_mask):
        # stack layers

        for l in range(self.num_unit):
            x, y = self.MCA_layer(x, y, x_mask, y_mask)
        return x, y

#----------------------------------------------------------------------------------------
#----------------------Att reduce module-------------------------------------------------
#-----------------------------------------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, hidden_size):
        super(AttFlat, self).__init__()
        self.hidden_size = hidden_size
        self.flat_mlp_size = 512
        self.flat_glimpses = 1
        self.drop_out = 0.1
        self.flat_out_size = 64
        self.mlp = MLP(
            in_size = self.hidden_size,
            mid_size = 64,#self.flat_mlp_size,
            out_size = self.flat_glimpses,
            dropout_r = self.drop_out,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            self.hidden_size * self.flat_glimpses,
            self.flat_out_size
        )


        ####################################################

    def forward(self, x, x_mask):


        att = self.mlp(x)

#---------------------------------------------------------------------------------
#----------------------mask mask mask --------------------------------------------
#---------------------------------------------------------------------------------

        # att = att.masked_fill(
        #     x_mask.squeeze(1).squeeze(1).unsqueeze(2),
        #     -1e9
        # )

        att = F.softmax(att, dim=1)


        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        # print(x_atted.shape,x.shape)
        x_atted = self.linear_merge(x_atted)

        return x_atted
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2