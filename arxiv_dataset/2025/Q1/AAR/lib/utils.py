import torch
import os
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib import rc
import torchvision.transforms.functional as F
import json

colors_list = np.loadtxt('./dataloader/color.txt', dtype=np.uint8)


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, gt_classes=None, correct_classes=None, did_not_predict_gt_class=None):

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # if gt_classes is not None and did_not_predict_gt_class is not None and correct_classes is not None:
        #     c_ = ', '.join(correct_classes)
        #     w_ = ', '.join(did_not_predict_gt_class)
        #     # print(c_)
        #     # print(w_)
        #     # input()
        #     axs[0,i].text(0.5,-0.08, c_, size=8, ha="center", transform=axs[0,i].transAxes, wrap=True, color='green')
        #     axs[0,i].text(0.5, -0.15, w_, size=8, ha="center", transform=axs[0,i].transAxes, wrap=True, color='red')
        #     # plt.xlabel
    plt.show()

def save(save_path, img_name, imgs, gt_classes=None, correct_classes=None, did_not_predict_gt_class=None, visualize=False):

    folder = img_name.split('/')[0]
    video_name = img_name.split('/')[1]
    img_name = img_name.split('/')[-1].split('.png')[0] + '_bbox.png'
    correct_classes_name = img_name.split('_bbox.png')[0] + '_correct.txt'
    did_not_predict_gt_class_name = img_name.split('_bbox.png')[0] + '_did_not_predict.txt'

    directory = os.path.join(save_path, folder + '/' + video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not visualize:
        img = Image.fromarray(imgs.permute(1,2,0).numpy())
        img.save(os.path.join(directory, img_name))

        np.savetxt(os.path.join(directory, correct_classes_name), correct_classes, delimiter=" ", fmt="%s")
        np.savetxt(os.path.join(directory, did_not_predict_gt_class_name), did_not_predict_gt_class, delimiter=" ", fmt="%s")
    else:
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()


def get_visualization_and_results(data_path, save_path, gt_annotation, AG_dataset, predicted_action_classes):

    gt_classes, correct_classes, did_not_predict_gt_class = get_predicted_actions(gt_annotation, predicted_action_classes)

    for i in range(len(gt_annotation)):
        obj_class = []
        obj_bbox = []

        for j in gt_annotation[i]:
            for k in j.keys():
                if k == 'person_bbox':
                    obj_class.append(AG_dataset.object_classes[1])
                    obj_bbox.append(j[k][0])
                elif k == 'class':
                    obj_class.append(AG_dataset.object_classes[j[k]])
                    obj_bbox.append(j['bbox'])
                elif k == 'metadata':
                    img_metadata = j[k]['tag'] + '.png'
                    vid_name = img_metadata.split('/')[0]
                    img_name = img_metadata.split('/')[-1]
                    full_name = 'frames/' + vid_name + '/' + img_name

        obj_bbox = torch.from_numpy(np.array(obj_bbox))

        img = read_image(os.path.join(data_path, full_name))

        colors = []

        for o_class in obj_class:
            colors.append(tuple(colors_list[AG_dataset.object_classes.index(o_class)]))

        result = draw_bounding_boxes(img, obj_bbox, colors=colors, labels=obj_class, width=2)

        save(save_path, full_name, result, gt_classes, correct_classes, did_not_predict_gt_class)
        # show(result, gt_classes, correct_classes, did_not_predict_gt_class)

def get_predicted_actions(gt_annotation, predicted_action_classes):
    with open('./Charades_annotations/Charades_v1_classes.txt', 'r') as f:
        charades_action_classes = f.readlines()

    actions = []
    for line in charades_action_classes:
        action_class = line[5:].split('\n')[0]
        actions.append(action_class)
    actions = np.array(actions)
    for i in range(len(gt_annotation)):
        # for multiple obj within an image

        gt_past_actions = np.array(gt_annotation[i][-1]['action_class'])
        top_k = len(gt_past_actions)
        predicted_past_actions = sorted(torch.topk(predicted_action_classes, k=top_k, dim=1)[1][0].cpu().numpy())
        correct = []
        model_predicted_wrongly = []

        correct = sorted(list(set(predicted_past_actions).intersection(set(gt_past_actions))))
        model_predicted_wrongly = sorted(list(set(predicted_past_actions) - set(gt_past_actions)))
        did_not_predict = sorted(list(set(gt_past_actions) - set(predicted_past_actions)))

        # print("The ground truth past action classes are: {}".format(actions[gt_past_actions]))
        # print("The model correctly predicted classes: {}".format(actions[correct]))
        # print("The model incorrectly predicted classes: {}".format(actions[model_predicted_wrongly]))
        # print("The model did not predict these classes: {}".format(actions[did_not_predict]))

        return actions[gt_past_actions], actions[correct], actions[did_not_predict]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_test_config(conf):
    with open(os.path.join(conf.conf_path, 'config.json'), 'r') as f:
        conf_dict = json.load(f)

    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)

    for i in conf_dict:
        if i == 'save_path' or i == 'model_path' or i == 'data_path' or i == 'conf_path' or i == 'infer_last' or i == 'visualize' or i == 'seq_model_path' or i == 'predict_flag':
            # do not load these from config file
            pass
        elif i not in conf.args:
            # check if any old config required
            conf.args[i] = conf_dict[i]
        else:
            # replace configurations with config file
            conf.args[i] = conf_dict[i]

    conf.__dict__.update(conf.args)

    if conf.model_type == 'transformer' or conf.model_type == 'GNNED' or conf.model_type == 'RBP' or conf.model_type == 'trans_RBP':
        conf.args.pop('emb_out')
        conf.args.pop('mlp_layers')
        conf.__dict__.update(conf.args)

    elif conf.model_type == 'mlp':
        conf.args.pop('enc_layer')
        conf.args.pop('dec_layer')
        conf.args.pop('emb_out')
        conf.__dict__.update(conf.args)
    else:
        conf.args.pop('mlp_layers')
        conf.__dict__.update(conf.args)

    if conf.task == 'set' or conf.task == 'verification':
        conf.args.pop('seq_model_path')
        conf.args.pop('seq_layer')
        conf.args.pop('seq_model')
        conf.args.pop('hidden_dim')
        conf.args.pop('seq_model_mlp_layers')
        conf.__dict__.update(conf.args)

    if conf.model_type != 'mlp' or conf.model_type != 'transformer':
        conf.args.pop('concept_net')

    for i in conf.args:
        if i == 'mode' or i =='ckpt' or i =='datasize' or i == 'resume':
            continue
        else:
            print(i,':', conf.args[i])

    return conf

def update_train_config(conf):
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)

    print('The config file is saved here:', conf.save_path)

    if conf.model_type == 'transformer' or conf.model_type == 'GNNED' or conf.model_type == 'RBP':
        conf.args.pop('emb_out')
        conf.args.pop('mlp_layers')
        conf.__dict__.update(conf.args)

    elif conf.model_type == 'mlp':
        conf.args.pop('enc_layer')
        conf.args.pop('dec_layer')
        conf.args.pop('emb_out')
        conf.__dict__.update(conf.args)
    elif conf.model_type == 'RBP':
        conf.args.pop('enc_layer')
        conf.args.pop('dec_layer')
        conf.__dict__.update(conf.args)
    else:
        conf.args.pop('mlp_layers')
        conf.__dict__.update(conf.args)

    if conf.task == 'set' or conf.task == 'verification':
        conf.args.pop('seq_model_path')
        conf.args.pop('seq_layer')
        conf.args.pop('seq_model')
        conf.args.pop('hidden_dim')
        conf.args.pop('seq_model_mlp_layers')
        conf.__dict__.update(conf.args)

    if conf.model_type == 'Relational':
        # conf.args.pop('mlp_layers')
        conf.args.pop('enc_layer')
        conf.args.pop('dec_layer')
        conf.args.pop('emb_out')
        conf.args.pop('nepoch')
        conf.args.pop('lr')
        conf.args.pop('optimizer')
        conf.args.pop('lr_scheduler')
        conf.args.pop('semantic')
        conf.args.pop('pool_type')
                
        conf.__dict__.update(conf.args)

    if conf.model_type != 'mlp' or conf.model_type != 'transformer':
        conf.args.pop('concept_net')

    for i in conf.args:
        if i == 'mode' or i =='ckpt' or i =='datasize' or i == 'resume' or i == 'infer_last':
            continue
        else:
            print(i,':', conf.args[i])

    with open(os.path.join(conf.save_path, 'config.json'), 'w') as f:
        json.dump(conf.args, f, indent=4)
        
    return conf 