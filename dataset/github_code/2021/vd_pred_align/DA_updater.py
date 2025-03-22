import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import convert
from chainercv.links.model.ssd import multibox_loss
from COWC_dataset_processed import vehicle_classes
from ops import  gradient_reversal_layer

class Adv_updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop('models')
        self.comm = kwargs.pop('comm')
        self.RTN_mbox = kwargs.pop('RTN_mbox')
        self.min_entropy = kwargs.pop('min_entropy')
        self.min_entropy_coef = kwargs.pop('min_entropy_coef')
        self.pred_align = kwargs.pop('pred_align')
        self.pred_align_weight = kwargs.pop('pred_align_weight')
        self.pred_align_mode = kwargs.pop('pred_align_mode')
        self.pred_align_norm_default_weight = kwargs.pop('pred_align_norm_default_weight')
        if self.pred_align_mode == 4:
            self.pred_weight_S = kwargs.pop('pred_align_norm_weight')
            self.pred_weight_T = self.pred_weight_S
        self.disable_pred_align_norm = kwargs.pop('disable_pred_align_norm')
        self.mbox_delta_inv = kwargs.pop("mbox_delta_inv")
        self.mbox_delta_inv_weight = kwargs.pop("mbox_delta_inv_weight")
        self.generator = kwargs.pop('generator')
        self.coGAN = kwargs.pop('coGAN')
        self.adv_inv = kwargs.pop('adv_inv')
        self.dis, self.cls = models
        self.buf = kwargs.pop('buffer')
        self.gpu_num = kwargs["device"]
        self.snapshot_interval = kwargs.pop('snapshot_interval')
        self.outdir = kwargs.pop('outdir')
        self.cls_batch_split = kwargs.pop('cls_batch_split')
        self.mboxloss_mode = kwargs.pop('mboxloss_mode')
        super(Adv_updater, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.xp = self.cls.xp
        if self.pred_align:
            if self.pred_align_mode in [2, 3]:
                self.pred_weight_S = self.calc_class_weight_source()
                self.pred_weight_T = self.calc_class_weight_target(init=True)
            else:
                self.pred_weight_S = None
                self.pred_weight_T = None

    def average_among_processes(self,value):
        if self.comm:
            value = self.comm.allreduce_obj(value) / self.comm.size
        return value

    def calc_norm_mask(self, mb_confs,mode, cls_weight = None, default_weight=None):
        class_num = mb_confs.shape[-1]
        if mode == 1:
            num_examples = np.prod(mb_confs.shape[:2])
            norm_num = num_examples / class_num
            if default_weight == None:
                default_weight = [1., ] * class_num
        pred_classes = self.xp.argmax(mb_confs.array, axis=-1)
        pred_classes = self.xp.expand_dims(pred_classes, axis=1)
        pred_classes = self.xp.expand_dims(pred_classes, axis=-1)
        norm_mask = self.xp.zeros(pred_classes.shape).astype("f")
        for c in range(class_num):
            mask_c = pred_classes == c
            if cls_weight == None:
                cls_weight = []
            if mask_c.sum() > 0:
                if mode == 1:
                    c_weight = norm_num / mask_c.sum() * default_weight[c]
                    norm_mask += mask_c.astype("f") * c_weight
                    cls_weight.append(c_weight)
                else:
                    norm_mask += mask_c.astype("f") * cls_weight[c]
        if self.iteration % 100 == 0 and (not self.comm or self.comm.rank == 0) and self.pred_align_mode == 1:
            print("cls_weight:{}".format(cls_weight))
        return norm_mask

    def calc_class_weight_source(self):
        source_iterator = self.get_iterator('main')
        num_class = len(vehicle_classes) + 1
        num_appear_S = np.array([0, ] * num_class)
        while not (source_iterator.is_new_epoch):
            batch_source = source_iterator.next()
            batch_source_array = convert.concat_examples(batch_source, self.device)
            batch_s_label = batch_source_array[2]
            for c in range(num_class):
                mask_c = batch_s_label == c
                num_appear_S[c] += mask_c.sum()
        if self.pred_align_mode == 2:
            class_weight_S = np.max(num_appear_S) / num_appear_S
        elif self.pred_align_mode == 3:
            class_weight_S = np.sum(num_appear_S) / num_class / num_appear_S
        source_iterator.reset()
        print("cls align weight (source): {}({} epoch)".format(class_weight_S,source_iterator.epoch))
        return class_weight_S

    def calc_class_weight_target(self,init=False):
        target_iterator = self.get_iterator('target')
        num_class = len(vehicle_classes) + 1
        num_appear_T = np.array([0, ] * num_class)
        batch_target = target_iterator.next()
        if self.t_rec_org == True:
            batch_target = [x[0] for x in batch_target]
        with chainer.no_backprop_mode():
            tgt_fmap = self.t_enc(Variable(self.xp.array(batch_target)))
            mb_locs_T, mb_confs_T = self.cls.multibox(tgt_fmap)
        pred_classes = self.xp.argmax(mb_confs_T.array, axis=-1)
        for c in range(num_class):
            mask_c = pred_classes == c
            num_appear_T[c] += mask_c.sum()
        while not (target_iterator.is_new_epoch):
            batch_target = target_iterator.next()
            if self.t_rec_org == True:
                batch_target = [x[0] for x in batch_target]
            with chainer.no_backprop_mode():
                tgt_fmap = self.t_enc(Variable(self.xp.array(batch_target)))
                mb_locs_T, mb_confs_T = self.cls.multibox(tgt_fmap)
            pred_classes = self.xp.argmax(mb_confs_T.array, axis=-1)
            for c in range(num_class):
                mask_c = pred_classes == c
                num_appear_T[c] += mask_c.sum()
        if self.pred_align_mode == 2:
            class_weight_T = np.max(num_appear_T) / num_appear_T
        elif self.pred_align_mode == 3:
            class_weight_T = np.sum(num_appear_T) / num_class / num_appear_T
        if init:
            target_iterator.reset()
        print("cls align weight (target): {}({} epoch)".format(class_weight_T,target_iterator.epoch))
        return class_weight_T

    def update_core(self):
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        if self.RTN_mbox:
            opt_mbox_delta = self.get_optimizer("opt_mbox_delta")
            mbox_delta = opt_mbox_delta.target
        if self.generator:
            gen_optimizer = self.get_optimizer('opt_gen')
        xp = self.cls.xp
        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.gpu_num)) if self.gpu_num >= 0 else lambda x: x

        source_iterator = self.get_iterator('main')
        batch_source = source_iterator.next()
        batch_target = self.get_iterator('target').next()
        if self.pred_align and self.pred_align_mode in [2, 3] and self.get_iterator('target').is_new_epoch:
            self.pred_weight_T = self.calc_class_weight_target()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        with chainer.no_backprop_mode():
            src_fmap = self.t_enc(batch_source_array[0])  # src feature map
            if self.generator:
                if self.coGAN:
                    src_fmap[0] += self.generator(batch_source_array[0])
        batchsize = len(batch_target)
        use_bufsize = int(batchsize/2)
        with chainer.no_backprop_mode():
            tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))

        #discriminator training for prediction alignment
        if self.pred_align:
            opt_cls_dis = self.get_optimizer('opt_cls_dis')
            cls_dis = opt_cls_dis.target
            cls_dis.cleargrads()
            mbox = self.cls.multibox
            with chainer.no_backprop_mode():
                src_fmap_ = src_fmap
                mb_locs_S, mb_confs_S = mbox(src_fmap_)
                if self.RTN_mbox:
                    src_fmap_delta = src_fmap_[:]
                    mb_locs_S_delta, mb_confs_S_delta = mbox_delta(src_fmap_delta)
                    mb_locs_S += mb_locs_S_delta
                    mb_confs_S += mb_confs_S_delta
                if self.pred_align_mode >= 1:
                    if not self.disable_pred_align_norm[0]:
                        norm_mask_S = self.calc_norm_mask(mb_confs_S[:, 0:4 * 38 * 38, :], self.pred_align_mode, self.pred_weight_S, self.pred_align_norm_default_weight)
                prediction_S = F.concat((mb_locs_S, mb_confs_S),axis=-1).transpose(0,2,1)
                prediction_S = prediction_S[:, :, 0:4 * 38 * 38]  # extract the part of the first feature map
                prediction_S = F.expand_dims(prediction_S,axis=-1)

                tgt_fmap_ = tgt_fmap
                mb_locs_T, mb_confs_T = mbox(tgt_fmap_)
                if self.pred_align_mode >= 1:
                    if not self.disable_pred_align_norm[1]:
                        norm_mask_T = self.calc_norm_mask(mb_confs_T[:, 0:4 * 38 * 38, :], self.pred_align_mode, self.pred_weight_T, self.pred_align_norm_default_weight)
                prediction_T = F.concat((mb_locs_T, mb_confs_T), axis=-1).transpose(0, 2, 1)
                prediction_T = prediction_T[:, :, 0:4 * 38 * 38]  # extract the part of the first feature map
                prediction_T = F.expand_dims(prediction_T, axis=-1)
            y_src_cls = cls_dis(prediction_S)
            y_tgt_cls = cls_dis(prediction_T)
            loss_dis_cls_src = F.softplus(-y_src_cls)
            loss_dis_cls_tgt = F.softplus(y_tgt_cls)
            if self.pred_align_mode >= 1:
                if not self.disable_pred_align_norm[0]:
                    loss_dis_cls_src *= norm_mask_S
                if not self.disable_pred_align_norm[1]:
                    loss_dis_cls_tgt *= norm_mask_T
            loss_dis_cls_src = F.sum(loss_dis_cls_src) / np.prod(y_src_cls.shape) # * self.cls_align_weight
            loss_dis_cls_tgt = F.sum(loss_dis_cls_tgt) / np.prod(y_tgt_cls.shape) # * self.cls_align_weight
            loss_dis_cls = loss_dis_cls_src + loss_dis_cls_tgt
            loss_dis_cls.backward()
            opt_cls_dis.update()
            loss_dis_cls.unchain_backward()
            loss_dis_cls_src = loss_dis_cls_src.data
            loss_dis_cls_tgt = loss_dis_cls_tgt.data
            loss_dis_cls = loss_dis_cls.data

        #discriminator training for plain adv
        buffer = self.buf
        discriminator = self.dis
        dis_opt = dis_optimizer
        s_fmap = src_fmap
        size = 0
        if batchsize >= 2:
            size, e_buf_src , e_buf_tgt = buffer.get_examples(use_bufsize)

        if size != 0:
            src_fmap_dis = []
            for i in range(len(s_fmap)):
                src_fmap_dis.append(F.vstack((F.copy(s_fmap[i][0:batchsize - size],self.gpu_num), Variable(func_bGPU(e_buf_src[i])))))
                src_fmap_dis[i].unchain_backward()
        else:
            src_fmap_dis = []
            for i in range(len(s_fmap)):
                src_fmap_dis.append(F.copy(s_fmap[i],self.gpu_num))
                src_fmap_dis[i].unchain_backward()

        y_source = discriminator(src_fmap_dis)

        tgt_fmap_dis = []
        for i in range(len(tgt_fmap)):
            tgt_fmap_dis.append(F.copy(tgt_fmap[i][0:batchsize-size],self.gpu_num))
            tgt_fmap_dis[i].unchain_backward()
            if size > 0:
                tgt_fmap_dis[i] = F.vstack([tgt_fmap_dis[i], Variable(func_bGPU(e_buf_tgt[i]))])

        y_target = discriminator(tgt_fmap_dis)

        n_fmap_elements = y_target.shape[2]*y_target.shape[3]

        loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        loss_dis = loss_dis_src + loss_dis_tgt

        discriminator.cleargrads()
        loss_dis.backward()
        dis_opt.update()

        loss_dis.unchain_backward()
        loss_dis = loss_dis.data
        loss_dis_src = loss_dis_src.data
        loss_dis_tgt = loss_dis_tgt.data
        del src_fmap_dis
        del tgt_fmap_dis

        #save fmap to buffer
        src_fmap_tobuf = []
        tgt_fmap_tobuf = []
        for i in range(len(s_fmap)):
            src_fmap_tobuf.append(chainer.cuda.to_cpu(s_fmap[i].data[:use_bufsize]))
            tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
        buffer.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

        batch_source = self.get_iterator('main').next()
        batch_target = self.get_iterator('target').next()
        if self.pred_align and self.pred_align_mode in [2, 3] and self.get_iterator('target').is_new_epoch:
            self.pred_weight_T = self.calc_class_weight_target()

        batch_source_array = convert.concat_examples(batch_source, self.device)

        self.cls.cleargrads()

        if self.RTN_mbox:
            mbox_delta.cleargrads()
        if self.generator:
            self.generator.cleargrads()

        loss_t_enc_sum = 0
        loss_cls_sum = 0
        loss_loc_sum = 0
        loss_conf_sum = 0

        if self.min_entropy:
            loss_t_entropy_sum = 0
        if self.pred_align:
            loss_dis_cls_enc_sum = 0
        if self.coGAN:
            loss_cls_org_sum = 0
        for b_num in range(-(-len(batch_source_array[0]) // self.cls_batch_split)):
            batch_split_s = batch_source_array[0][
                          self.cls_batch_split * b_num:self.cls_batch_split * (b_num + 1)]
            batch_split_s_loc = batch_source_array[1][self.cls_batch_split * b_num:self.cls_batch_split * (b_num + 1)]
            batch_split_s_label = batch_source_array[2][
                                self.cls_batch_split * b_num:self.cls_batch_split * (b_num + 1)]
            batch_split_t = batch_target[self.cls_batch_split * b_num:self.cls_batch_split * (b_num + 1)]
            split_coef = len(batch_split_s) / len(batch_source_array[0])
            split_coef_ = len(batch_split_s) / (len(batch_source_array[0]))
            s_data = Variable(batch_split_s)
            t_data = Variable(xp.array(batch_split_t))

            src_fmap = self.t_enc(s_data)  # src feature map
            tgt_fmap = self.t_enc(t_data)

            if self.coGAN: # coGAN_DA training
                #supervised training without augmentation
                multibox = self.cls.multibox
                src_fmap_ = [src_fmap[i] + 0 for i in range(len(src_fmap))]
                mb_locs, mb_confs = multibox(src_fmap_)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, batch_split_s_loc, batch_split_s_label, self.k, comm=self.comm)
                loc_loss *= split_coef_
                conf_loss *= split_coef_
                cls_loss = loc_loss * self.alpha + conf_loss

                cls_loss.backward()

                loss_cls_org_sum += cls_loss.data

                for fmap in src_fmap_:
                    fmap.unchain()
                cls_loss.unchain_backward()

                #augumentation of source domain features
                src_fmap[0] += self.generator(s_data)

            if self.adv_inv: # coGAN feature alignment
                dis = self.dis
                _fmap_adv = src_fmap
                fmap_adv = [_fmap_adv[i] + 0 for i in range(len(_fmap_adv))]
                y_target_enc = dis(fmap_adv)
                loss_t_enc = F.sum(F.softplus(y_target_enc)) / n_fmap_elements / batchsize * split_coef
                loss_t_enc.backward()
                for fmap in fmap_adv:
                    fmap.unchain()
                loss_t_enc.unchain_backward()
            else: #feature alignment of plain adv
                fmap_adv = tgt_fmap
                if self.min_entropy or self.pred_align:
                    fmap_adv = [fmap_adv[i] + 0 for i in range(len(fmap_adv))]
                y_target_enc = self.dis(fmap_adv)
                loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize * split_coef
                loss_t_enc.backward()
                if self.min_entropy or self.pred_align:
                    for fmap in fmap_adv:
                        fmap.unchain()
                loss_t_enc.unchain_backward()
            loss_t_enc_sum += loss_t_enc.data

            # supervised training on source domain
            mbox = self.cls.multibox
            mb_locs, mb_confs = mbox(src_fmap)
            if self.RTN_mbox:
                src_fmap_ = src_fmap[:]
                mb_locs_delta, mb_confs_delta = mbox_delta(src_fmap_)
                mb_locs += mb_locs_delta
                mb_confs += mb_confs_delta
            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, batch_split_s_loc, batch_split_s_label, self.k,comm=self.comm)
            loc_loss *= split_coef_
            conf_loss *= split_coef_
            cls_loss = loc_loss * self.alpha + conf_loss #cls loss

            cls_loss.backward()

            loss_cls_sum += cls_loss.data
            loss_loc_sum += loc_loss.data
            loss_conf_sum += conf_loss.data

            cls_loss.unchain_backward()

            if self.pred_align: # prediction alignment
                mbox_targets = ["original"]
                if self.RTN_mbox and self.mbox_delta_inv:
                    mbox_targets.append("mbox_delta_inv")
                for mbox_target in mbox_targets:
                    tgt_fmap_ = tgt_fmap[:]
                    for i in range(len(tgt_fmap_)):
                        tgt_fmap_[i] = tgt_fmap_[i] + 0
                    if self.RTN_mbox and mbox_target == "mbox_delta_inv":
                        tgt_fmap_delta = tgt_fmap_[:]
                    mbox = self.cls.multibox
                    if mbox_target in ["mbox_delta_inv"] and not self.mbox_delta_inv in [5, 6, 7, 8]: chainer.config.enable_backprop = False
                    mb_locs_T, mb_confs_T = mbox(tgt_fmap_)
                    if mbox_target in ["mbox_delta_inv"] and self.mbox_delta_inv in [6, 8]:
                        for i in range(len(tgt_fmap_)):
                            tgt_fmap_[i].unchain()
                    if self.RTN_mbox and mbox_target == "mbox_delta_inv":
                        if mbox_target == "mbox_delta_inv":
                            if not self.mbox_delta_inv in [1, 4]: chainer.config.enable_backprop = True
                            tgt_fmap_delta_ = tgt_fmap_delta[:]
                            for i in range(len(tgt_fmap_delta_)):
                                tgt_fmap_delta_[i] = tgt_fmap_delta_[i] + 0
                            chainer.config.enable_backprop = True
                            mb_locs_delta, mb_confs_delta = mbox_delta(tgt_fmap_delta_)
                            mb_locs_T += mb_locs_delta
                            mb_confs_T += mb_confs_delta
                        else:
                            mb_locs_delta, mb_confs_delta = mbox_delta(tgt_fmap_delta)
                            mb_locs_T += mb_locs_delta
                            mb_confs_T += mb_confs_delta
                    if self.pred_align_mode >= 1:
                        if not self.disable_pred_align_norm[2]:
                            norm_mask_T = self.calc_norm_mask(mb_confs_T[:, 0:4 * 38 * 38, :], self.pred_align_mode, self.pred_weight_T, self.pred_align_norm_default_weight)
                    prediction_T = F.concat((mb_locs_T, mb_confs_T), axis=-1).transpose(0, 2, 1)
                    prediction_T = prediction_T[:, :, 0:4 * 38 * 38]  # extract the part of the first feature map
                    prediction_T = F.expand_dims(prediction_T, axis=-1)

                    y_tgt_cls_enc = cls_dis(prediction_T)
                    if mbox_target in ["mbox_delta_inv"] and self.mbox_delta_inv in [3, 4, 7, 8]:
                        loss_dis_cls_enc = F.softplus(y_tgt_cls_enc)
                    else:
                        loss_dis_cls_enc = F.softplus(-y_tgt_cls_enc)
                    if self.pred_align_mode >= 1:
                        if not self.disable_pred_align_norm[2]:
                            loss_dis_cls_enc *= norm_mask_T
                    loss_dis_cls_enc = F.sum(loss_dis_cls_enc) / np.prod(y_tgt_cls_enc.shape) * self.pred_align_weight * split_coef_
                    if mbox_target in ["mbox_delta_inv"]:
                        if self.mbox_delta_inv in [1, 2, 5, 6]:
                            loss_dis_cls_enc, = gradient_reversal_layer(loss_dis_cls_enc)
                        loss_dis_cls_enc *= self.mbox_delta_inv_weight
                    loss_dis_cls_enc.backward()
                    for i in range(len(tgt_fmap_)):
                        tgt_fmap_[i].unchain()
                    loss_dis_cls_enc.unchain_backward()
                    if mbox_target == "original":
                        loss_dis_cls_enc_sum += loss_dis_cls_enc.data

            if self.min_entropy: #entropy minimization
                mb_locs, mb_confs = self.cls.multibox(tgt_fmap)
                mb_confs = mb_confs[:, 0:4 * 38 * 38, :] # extract the part of the first feature map
                mb_confs_p = F.softmax(mb_confs, axis=-1)
                if self.min_entropy == 1:
                    norm_coef = np.prod(mb_confs_p.shape[:2])
                else:
                    norm_coef = mb_confs_p.shape[0]
                entropy = - F.sum(mb_confs_p * F.log2(mb_confs_p + (mb_confs_p.array < 1e-30) * 1e-30))/ norm_coef * split_coef_ * self.min_entropy_coef
                loss_t_entropy_sum += entropy.data
                entropy.backward()
                entropy.unchain_backward()

        if self.coGAN:
            gen_optimizer.update()

        cls_optimizer.update()
        if self.RTN_mbox:
            opt_mbox_delta.update()
        self.cls.cleargrads()

        loss_cls_sum =  self.average_among_processes(loss_cls_sum)
        loss_loc_sum = self.average_among_processes(loss_loc_sum)
        loss_conf_sum = self.average_among_processes(loss_conf_sum)

        if self.min_entropy:
            loss_t_entropy_sum = self.average_among_processes(loss_t_entropy_sum)

        if self.generator and self.coGAN:
            loss_cls_org_sum = self.average_among_processes(loss_cls_org_sum)

        loss_t_enc_sum = self.average_among_processes(loss_t_enc_sum)
        loss_dis = self.average_among_processes(loss_dis)
        loss_dis_src = self.average_among_processes(loss_dis_src)
        loss_dis_tgt = self.average_among_processes(loss_dis_tgt)
        if self.pred_align:
            loss_dis_cls_enc_sum = self.average_among_processes(loss_dis_cls_enc_sum)
            loss_dis_cls = self.average_among_processes(loss_dis_cls)
            loss_dis_cls_src = self.average_among_processes(loss_dis_cls_src)
            loss_dis_cls_tgt = self.average_among_processes(loss_dis_cls_tgt)

        if not self.comm or self.comm.rank == 0:
            chainer.reporter.report({'loss_t_enc':loss_t_enc_sum})
            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_dis_src': loss_dis_src})
            chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt})
            if self.pred_align:
                chainer.reporter.report({'loss_dis_cls_enc': loss_dis_cls_enc_sum})
                chainer.reporter.report({'loss_dis_cls': loss_dis_cls})
                chainer.reporter.report({'loss_dis_cls_src': loss_dis_cls_src})
                chainer.reporter.report({'loss_dis_cls_tgt': loss_dis_cls_tgt})
            chainer.reporter.report({'loss_cls': loss_cls_sum})
            chainer.reporter.report({'loss_loc': loss_loc_sum})
            chainer.reporter.report({'loss_conf': loss_conf_sum})
            if self.min_entropy:
                chainer.reporter.report({'loss_t_entropy': loss_t_entropy_sum})
            if self.generator and self.coGAN: # and self.adv_mutual != "separate_alt":
                # if self.coGAN_s_train:
                chainer.reporter.report({'loss_cls_org': loss_cls_org_sum})

    def serialize(self, serializer):
        super().serialize(serializer)
        self.buf.serialize(serializer['buf'])