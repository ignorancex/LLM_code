# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import torch


import numpy as np

category_dict = {
    'voc': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
            'dog',
            'horse', 'motorbike', 'player', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'],
    'coco': ['person',
                   'bicycle',
                   'car',
                   'motorcycle',
                   'airplane',
                   'bus',
                   'train',
                   'truck',
                   'boat',
                   'traffic light',
                   'fire hydrant',
                   'street sign',
                   'stop sign',
                   'parking meter',
                   'bench',
                   'bird',
                   'cat',
                   'dog',
                   'horse',
                   'sheep',
                   'cow',
                   'elephant',
                   'bear',
                   'zebra',
                   'giraffe',
                   'hat',
                   'backpack',
                   'umbrella',
                   'shoe',
                   'eye glasses',
                   'handbag',
                   'tie',
                   'suitcase',
                   'frisbee',
                   'skis',
                   'snowboard',
                   'sports ball',
                   'kite',
                   'baseball bat',
                   'baseball glove',
                   'skateboard',
                   'surfboard',
                   'tennis racket',
                   'bottle',
                   'plate',
                   'wine glass',
                   'cup',
                   'fork',
                   'knife',
                   'spoon',
                   'bowl',
                   'banana',
                   'apple',
                   'sandwich',
                   'orange',
                   'broccoli',
                   'carrot',
                   'hot dog',
                   'pizza',
                   'donut',
                   'cake',
                   'chair',
                   'couch',
                   'potted plant',
                   'bed',
                   'mirror',
                   'dining table',
                   'window',
                   'desk',
                   'toilet',
                   'door',
                   'tv',
                   'laptop',
                   'mouse',
                   'remote',
                   'keyboard',
                   'cell phone',
                   'microwave',
                   'oven',
                   'toaster',
                   'sink',
                   'refrigerator',
                   'blender',
                   'book',
                   'clock',
                   'vase',
                   'scissors',
                   'teddy bear',
                   'hair drier',
                   'toothbrush']

}

background_dict = {
    'voc': ['a photo of tree.', 'a photo of river.',
            'a photo of sea.', 'a photo of lake.', 'a photo of water.',
            'a photo of railway.', 'a photo of railroad.', 'a photo of track.',
            'a photo of stone.', 'a photo of rocks.'],
    'coco': ['a photo of street sign.', 'a photo of mountain.', 'a photo of video game.', 'a photo of men.',
             'a photo of track.', 'a photo of bus stop.', 'a photo of cabinet.', 'a photo of tray.',
             'a photo of plate.', 'a photo of shirt.', 'a photo of city street.', 'a photo of runway.',
             'a photo of tower.', 'a photo of ramp.', 'a photo of grass.', 'a photo of pillow.',
             'a photo of urinal.', 'a photo of lake.', 'a photo of brick.', 'a photo of fence.',
             'a photo of shower.', 'a photo of airport.', 'a photo of animal.', 'a photo of shower curtain.',
             'a photo of road.', 'a photo of mirror.', 'a photo of jacket.', 'a photo of church.', 'a photo of snow.',
             'a photo of fruit.', 'a photo of hay.', 'a photo of floor.', 'a photo of field.', 'a photo of street.',
             'a photo of mouth.', 'a photo of steam engine.', 'a photo of cheese.', 'a photo of river.',
             'a photo of tree branch.', 'a photo of suit.', 'a photo of child.', 'a photo of soup.', 'a photo of desk.',
             'a photo of tub.', 'a photo of tennis court.', 'a photo of teeth.', 'a photo of bridge.',
             'a photo of sky.', 'a photo of officer.', 'a photo of sidewalk.', 'a photo of dock.',
             'a photo of tree.', 'a photo of court.', 'a photo of rock.', 'a photo of board.',
             'a photo of branch.', 'a photo of pan.', 'a photo of box.', 'a photo of body.',
             'a photo of salad.', 'a photo of dirt.', 'a photo of leaf.', 'a photo of hand.',
             'a photo of highway.', 'a photo of vegetable.', 'a photo of computer monitor.',
             'a photo of door.', 'a photo of meat.', 'a photo of pair.', 'a photo of beach.',
             'a photo of harbor.', 'a photo of ocean.', 'a photo of baseball player.', 'a photo of girl.',
             'a photo of market.', 'a photo of window.', 'a photo of blanket.', 'a photo of boy.', 'a photo of woman.',
             'a photo of bat.', 'a photo of baby.', 'a photo of flower.', 'a photo of wall.', 'a photo of bath tub.',
             'a photo of tarmac.', 'a photo of tennis ball.', 'a photo of roll.', 'a photo of park.'],
}

prompt_dict = ['a photo of {}.']


def to_text(labels, dataset='voc'):
    _d = category_dict[dataset]

    text = []
    for i in range(labels.size(0)):
        idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
        if torch.sum(labels[i]) == 1:
            idx = idx.unsqueeze(0)
        cnt = idx.shape[0] - 1
        if cnt == -1:
            text.append('background')
        elif cnt == 0:
            text.append(prompt_dict[cnt].format(_d[idx[0]]))
        elif cnt == 1:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]]))
        elif cnt == 2:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]]))
        elif cnt == 3:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]]))
        elif cnt == 4:
            text.append(prompt_dict[cnt].format(_d[idx[0]], _d[idx[1]], _d[idx[2]], _d[idx[3]], _d[idx[4]]))
        else:
            raise NotImplementedError
    return text


import clip
def clip_forward(clip_model, images, labels, dname='coco'):
    texts = to_text(labels, dname)
    texts = clip.tokenize(texts).cuda()

    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    text_features = text_features.reshape(N, C, 1)

    similarity = torch.matmul(image_features, text_features)

    return similarity


def clip_forward_global(clip_model, images, labels, btp, dname='coco'):

    template_tokens_dict = clip.get_template_tokens_dict()
    btp_features = clip_model.encode_btp(btp, template_tokens_dict)
    btp_features = btp_features[labels.argmax(dim=1)]
    btp_features = btp_features / btp_features.norm(dim=-1, keepdim=True)


    image_features = clip_model.encode_image(images)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)


    label_features = clip_model.encode_text(clip.tokenize(to_text(labels, dname)).cuda())
    label_features = label_features / label_features.norm(dim=-1, keepdim=True)



    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    label_features = label_features.reshape(N, 1, C)
    btp_features = btp_features.reshape(N, C, 1)

    img_similarity = torch.matmul(image_features, btp_features)
    label_similarity = torch.matmul(label_features, btp_features)

    return img_similarity, label_similarity


def clip_forward_vis(clip_model, images, labels, btp, dname='coco'):

    template_tokens_dict = clip.get_template_tokens_dict()
    btp_features = clip_model.encode_btp(btp, template_tokens_dict)
    btp_features = btp_features[labels.argmax(dim=1)]
    btp_features = btp_features / btp_features.norm(dim=-1, keepdim=True)


    image_features = clip_model.encode_image_all(images)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    N, L, C = image_features.size()
    btp_features = btp_features.reshape(N, C, 1)

    similarity_maps = torch.matmul(image_features, btp_features).squeeze().reshape(N, 1, int(L**0.5), int(L**0.5)).clamp(min=0.0)


    return similarity_maps


def clip_forward_local(clip_model, images, labels, btp, dname='coco'):

    template_tokens_dict = clip.get_template_tokens_dict()
    btp_features = clip_model.encode_btp(btp, template_tokens_dict)
    btp_features = btp_features[labels.argmax(dim=1)]
    btp_features = btp_features / btp_features.norm(dim=-1, keepdim=True)


    image_features = clip_model.encode_image_all(images)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)


    label_features = clip_model.encode_text(clip.tokenize(to_text(labels, dname)).cuda())
    label_features = label_features / label_features.norm(dim=-1, keepdim=True)



    N, L, C = image_features.size()
    # image_features = image_features.reshape(N, 1, C)
    label_features = label_features.reshape(N, 1, C)
    btp_features = btp_features.reshape(N, C, 1)

    img_similarity = torch.matmul(image_features, btp_features).max(dim=1, keepdim=True)[0]
    label_similarity = torch.matmul(label_features, btp_features)

    return img_similarity, label_similarity


def clip_forward_btp2text(clip_model, btp, dname='coco'):


    texts = []
    with open('nouns.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            texts.append(prompt_dict[0].format(line))


    template_tokens_dict = clip.get_template_tokens_dict()
    btp_features = clip_model.encode_btp(btp, template_tokens_dict)
    btp_features = btp_features / btp_features.norm(dim=-1, keepdim=True)

    # print(torch.matmul(btp_features, btp_features.t()))


    text_features = clip_model.encode_text(clip.tokenize(texts).cuda())
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = text_features.size()

    similarities = torch.matmul(btp_features, text_features.t())
    texts = np.array(texts)
    retrieved_texts = texts[similarities.argmax(dim=1).cpu().numpy()]

    for category, retrieved_text, similarity in zip(category_dict[dname], retrieved_texts, similarities.max(dim=1)[0]):
        print(category, retrieved_text, similarity.item())

    return


def clip_forward_vis2(clip_model, image, text):

    image_feature = clip_model.encode_image_all(image)
    # normalized features
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

    text_feature = clip_model.encode_text(clip.tokenize([text]).cuda()).cuda()

    N, L, C = image_feature.size()
    text_feature = text_feature.reshape(N, C, 1)

    similarity_map = torch.matmul(image_feature, text_feature).squeeze().reshape(N, 1, int(L**0.5), int(L**0.5)).clamp(min=0.0)


    return similarity_map