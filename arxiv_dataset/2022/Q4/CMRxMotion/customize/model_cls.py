from abc import ABCMeta,abstractmethod
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from framework_pytorch.utils import loss_fuctions as LF, eval_function as EF, utils as U


""" Abstract class
    Custom model should inherit this class
    function train_step and eval_step must be defined

"""
class Model(metaclass=ABCMeta):
    def __init__(self, net, optimizer, device):
        
        self.net = net
        self.optimizer = optimizer
        self.device = device

        # move
        nets = net if type(net) in [list, tuple] else [net]
        [n.to(device) for n in nets]
        

    @abstractmethod
    def train_step(self, batch, epoch):
        """
        """

    @abstractmethod
    def eval_step(self, batch, **kwargs):
        """
        """

class ClsModel(Model):

    # init function, custom according to requirements
    def __init__(self, net, optimizer, device, img_suffix, lab_suffix, loss_functions={LF.CrossEntropy(): 0.5, LF.SoftDice(): 0.5}, regularzation=None):
        super().__init__(net, optimizer, device)
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.loss_function = loss_functions
        self.regularzation = regularzation

    # TODO: must be defined for training. 
    # network forward and backward need to be done here.
    def train_step(self, batch, epoch):
        img = batch[self.img_suffix]
        img = img.to(device=self.device)

        # switch to train model, zero gradient for optimizer
        self.net.train()
        self.optimizer.zero_grad()

        # forward + backward + optimize
        logits = self.net(img)
        loss = self.get_loss(batch, logits)   # calculate loss
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy() # return loss for logging

    # loss calculation, can be modified as required
    def get_loss(self, batch, logits):
        lab = batch[self.lab_suffix]
        lab = lab.to(device=self.device)
        loss = 0
        for lf in self.loss_function:
            loss += lf(logits, lab) * self.loss_function[lf]

        if self.regularzation is not None:
            loss += self.regularzation(self.net.parameters())

        return loss

    # TODO: must be defined for evaluation.
    # evaluate and return results dict {metric1: value, metric2: [value] ...}
    def eval_step(self, batch, **kwargs):
        self.net.eval() # switch to evaluation model

        img = batch[self.img_suffix]
        lab = batch[self.lab_suffix]
        img_g = img.to(device=self.device)
        lab_g = lab.to(device=self.device)

        with torch.no_grad():
            logits = self.net(img_g)
            loss = self.get_loss(batch, logits).cpu().numpy()

        pred = F.softmax(logits, dim=1)
        pred = torch.argmax(pred, -1).cpu().numpy()
        lab_g = torch.argmax(lab_g, -1).cpu().numpy()
        acc = EF.accuracy(pred, lab_g)
        acc_s = np.mean([pred[lab_g==0]==0, pred[lab_g==1]==1, pred[lab_g==2]==2], -1)

        # pred_vote = np.array([np.bincount(pred[:10]).argmax(), np.bincount(pred[10:20]).argmax(), np.bincount(pred[20:]).argmax()])
        # lab_vote = np.array([np.bincount(lab_g[:10]).argmax(), np.bincount(lab_g[10:20]).argmax(), np.bincount(lab_g[20:]).argmax()])
        # acc_vote = EF.accuracy(pred_vote, lab_vote)
        eval_results = {'loss': loss,
                        'acc': acc,
                        'acc_class': [acc_s]}
                        # 'acc_vote': acc_vote}
        
        # generate image results for logger and visulization
        # the key of image results must be 'image' or 'img'
        log_image = kwargs.get('log_image', False)
        if log_image:
            # tensor to numpy
            img = img.numpy()
            img = self.get_imgs_eval([img, lab_g, pred]) # create drawable image, see more details inside the function
            eval_results.update({'image': img})

        # TODO: custom argument for trainer.test(), for example:
        # in main code,trainer.test(..., need_logits=True) was called
        # this file add:
        # need_logits = kwargs.get('need_logits', False)
        # eval_results.update({'logits': logits.cpu().numpy()})

        return eval_results
        

    def get_imgs_eval(self, imgs):
        # check the images are 2d or 3d
        if len(np.shape(imgs[0])) == 4:
            return self.get_imgs_eval_2d(imgs)
        elif len(np.shape(imgs[0])) == 5:
            return self.get_imgs_eval_3d(imgs)
        else:
            raise('Error: image dimension should be 4 (2d) or 5 (3d)!')

    # TODO: modify as required
    # return dict {'name': [drawable image]}
    def get_imgs_eval_2d(self, imgs):
        xs, ys, prob = imgs
        prob = np.array(prob)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # ys_img = xs[0][0].copy()
        # ys_img = U.gray2rgb(U.recale_array(ys_img))
        ys_img = np.zeros(list(xs[0][0].shape) + [3])
        cv2.putText(ys_img, 'gt {:d}'.format(ys[0]), (30, 100), font, 2, (255, 0, 0), 2)
        cv2.putText(ys_img, 'pr {:d}'.format(prob[0]), (30, 180), font, 2, (255, 0, 0), 2)
        if xs.shape[1] == 2:
            img = U.combine_2d_imgs_from_tensor([xs[0:1,0,...], xs[0:1,1,...], np.expand_dims(ys_img, 0)])
        else:
            img = U.combine_2d_imgs_from_tensor([xs[0:1,0,...], np.expand_dims(ys_img, 0)])
        # img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        img_dict = {'cls': img}
        return img_dict


    def get_imgs_eval_3d(self, imgs):
        img, lab, pred = imgs
        # for 3d image, only visulize [vis_slices] slices, image shape should be [B,C,H,W,D], otherwise [B,H,W,D,C]
        img = U.combine_3d_imgs_from_tensor([img, lab, pred], vis_slices=6, channel_first=True)
        img_dict = {'img-lab-pred': img}
        return img_dict
