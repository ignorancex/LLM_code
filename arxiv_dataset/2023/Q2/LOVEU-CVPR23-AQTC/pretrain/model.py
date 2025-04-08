import torch
from torch import nn
from mmpt.models import MMPTModel
from s3dg import S3D
from pytorch_lightning import LightningModule
import timm, os
from transformers import AutoModel
import numpy as np

class VideoCLIP(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        model, tokenizer, aligner = MMPTModel.from_pretrained(
            "/data/zclfe/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml",
            '/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/pretrain/output/videoclip/checkpoint_best.pt')
        
        self.model = model
        self.tokenizer = tokenizer
        self.aligner = aligner
        caps, cmasks = aligner._build_text_seq(
            tokenizer("template", add_special_tokens=False)["input_ids"]
        )
        self.caps = caps.unsqueeze(0).cuda()
        self.cmasks = cmasks.unsqueeze(0).cuda()
        self.for_video = cfg.FOR.VIDEO
        self.for_script = cfg.FOR.SCRIPT
        self.for_qa = cfg.FOR.QA
        self.for_para = cfg.FOR.PARA
        self.suffix = cfg.SUFFIX if cfg.SUFFIX else ''

    def test_step(self, batch, idx):
        if batch[0] is None:
            return 

        if self.for_script:
            script, timestamps, video_frame, path = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            features = torch.cat([self.text_model(**sentence).last_hidden_state[:,-1,:] for sentence in script])
            torch.save([timestamps, features], os.path.join(path, "script.pth"))

        if self.for_para:
            timestamps, paras, video_frame, path = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            tokens = self.tokenizer([s for s in paras], add_special_tokens=False)["input_ids"]
            tokens = [self.aligner._build_text_seq(token) for token in tokens]
            caps, cmasks = zip(*tokens)
            caps = torch.stack(caps,0).cuda()
            cmasks = torch.stack(cmasks,0).cuda()
            features = []
            for i in range(caps.shape[0]):
                features.append(self.model.forward_text(caps[i].unsqueeze(0),cmasks[i].unsqueeze(0)))
            features = torch.cat(features,0)
            torch.save([timestamps, features], os.path.join(path, "paras.pth"))

            para_video = []
            video_frame = video_frame.cuda()

            for start_t,end_t in timestamps:
                para_video_feature = self.model.forward_video(video_frame[start_t:end_t].unsqueeze(0),self.caps,self.cmasks)
                para_video.append(para_video_feature)
            para_video = torch.cat(para_video,0)
            torch.save(para_video, os.path.join(path, "paras_video_features.pth"))


        if self.for_qa:
            qas, path, tag = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            for qa in qas:
                caps, cmasks = self.aligner._build_text_seq(
                    self.tokenizer(qa['question'], add_special_tokens=False)["input_ids"]
                )
                caps, cmasks = caps[None, :], cmasks[None, :]
                caps = caps.cuda()
                cmasks = cmasks.cuda()
                qa['question'] = self.model.forward_text(caps,cmasks)
                button_features = []
                for button_images_per_step in qa['button_images']:
                    button_features.append(
                        [
                            self.model.forward_button(button_image,self.caps,self.cmasks) \
                            for button_image in button_images_per_step
                        ]
                    )
                for i, answers_per_step in enumerate(qa['answers']):
                    for j, answer in enumerate(answers_per_step):
                        bidx = qa['answer_bidxs'][i][j]
                        button_feature = button_features[i][bidx]

                        caps, cmasks = self.aligner._build_text_seq(
                            self.tokenizer(answer, add_special_tokens=False)["input_ids"]
                        )
                        caps, cmasks = caps[None, :], cmasks[None, :]
                        caps = caps.cuda()
                        cmasks = cmasks.cuda()

                        text_feature = self.model.forward_text(caps,cmasks)
                        answer_feature = dict(text=text_feature, button=button_feature)
                        qa['answers'][i][j] = answer_feature
            torch.save(qas, os.path.join(path, f'{tag}{self.suffix}.pth'))


class S3DModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.model = S3D('/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/pretrain/output/videoclip/s3d_dict.npy',512)
        self.model.load_state_dict(torch.load('/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/pretrain/output/videoclip/s3d_howto100m.pth'))
        
        self.for_video = cfg.FOR.VIDEO
        self.for_script = cfg.FOR.SCRIPT
        self.for_qa = cfg.FOR.QA
        self.for_para = cfg.FOR.PARA
        self.suffix = cfg.SUFFIX if cfg.SUFFIX else ''
        self.batch_size = 16

    def test_step(self, batch, idx):
        if batch[0] is None:
            return 
        
        if self.for_video:
            video, path = batch[0] 
            # set smaller batch size to prevent OOM
            T = video.shape[0]
            if T%30:
                pad_len = 30*(T//30+1)-T
                padding = torch.zeros((pad_len,3,224,224)).cuda()
                video = torch.cat([video,padding],dim=0)
            T = int(video.shape[0]/30)
            video = video.view(T,30,3,224,224)
            video = video.permute(0,2,1,3,4)

            with torch.no_grad():
                num_batches = int(np.ceil(T / self.batch_size))
                outputs = []
                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, T)
                    batch_X = video[start_idx:end_idx]
                    batch_output = self.model(batch_X)['video_embedding']
                    outputs.append(batch_output)
                outputs = torch.cat(outputs,0)
            

            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(outputs, os.path.join(path, "video.pth"))

def build_model(cfg):
    return eval(cfg.MODEL)(cfg)














def get_model(type):
    return eval(type)