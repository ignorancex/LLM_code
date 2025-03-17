import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys, argparse, os, copy, glob
import numpy as np
import time
import random
import json

from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from Models.DTFD.network import get_cam_1d

torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

def get_slides_info(slides: dict, num_classes: int):
    slide_names = []
    slide_feats = []
    slide_labels = []
    for (slide_feats_path, slide_label) in slides.items():
        slide_names.append(Path(slide_feats_path).stem)
        feats = torch.load(slide_feats_path) # N x feats_dim
        slide_feats.append(feats)

        one_hot_label = np.zeros(num_classes)
        if num_classes == 1:
            one_hot_label[0] = slide_label
        else:
            if slide_label <= (len(one_hot_label)-1):
                one_hot_label[slide_label] = 1
        slide_labels.append(one_hot_label)
    
    return slide_names, slide_feats, slide_labels

def trainDTFD(args, train_slides_info_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, epoch, \
        criterion=None, numGroup=4, total_instance=4, log_path=''):
    # train_slides_info_list: [names, feats, labels]
    slide_names, slide_feats, slide_labels = train_slides_info_list
    distill = args.distill

    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0

    classifier.train()
    if not args.weight_path:
        dimReduction.train()
    else:
        dimReduction.eval()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    numSlides = len(slide_names)
    numIter = numSlides // args.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        tidx_slide = tIDX[idx * args.batch_size:(idx + 1) * args.batch_size]
        tslide_name = [slide_names[sst] for sst in tidx_slide]
        tlabel = [slide_labels[sst] for sst in tidx_slide]

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslide_feats = slide_feats[slide_idx].to(args.device)
            # shuffle the slide features (based on the rows/position of patch features arranged)
            tslide_feats = tslide_feats[torch.randperm(tslide_feats.shape[0])]
            tslide_label = torch.from_numpy(tlabel[tidx]).unsqueeze(0).to(args.device)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = tslide_feats

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslide_label)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(args.device))
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict, bg_feat0, Att_s0 = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                # af_inst_feat = tattFeat_tensor
                af_inst_feat = bg_feat0

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
            grad_clipping = 5.0
            if optimizer0:
                optimizer0.zero_grad()
                loss0.backward(retain_graph=True)
                if not args.weight_path:
                    torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
                torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)
                optimizer0.step()

            ## optimization for the second tier
            gSlidePred, bg_feat, Att_s1 = UClassifier(slide_pseudo_feat.detach())
            # gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = criterion(gSlidePred, tslide_label).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), grad_clipping)
            optimizer1.step()
            total_loss = total_loss + loss0.item() + loss1.item()

            UAtt = Att_s1
            atten_max = atten_max + UAtt.max().item()
            atten_min = atten_min + UAtt.min().item()
            atten_mean = atten_mean +  UAtt.mean().item()

            sys.stdout.write('\r Training slide [{:}/{:}] slide loss: {:.4f}, attention max:{:.5f}, min:{:.5f}, mean:{:.5f}'.\
                format(idx, len(slide_names), loss0.item() + loss1.item(), UAtt.max().item(), UAtt.min().item(), UAtt.mean().item()))

    atten_max = atten_max / len(slide_names)
    atten_min = atten_min / len(slide_names)
    atten_mean = atten_mean / len(slide_names)

    with open(log_path,'a+') as log_txt:
            log_txt.write('\n atten_max'+str(atten_max))
            log_txt.write('\n atten_min'+str(atten_min))
            log_txt.write('\n atten_mean'+str(atten_mean))

    return total_loss / len(slide_names)


def testDTFD(args, test_slides_info_list, classifier, dimReduction, attention, UClassifier, \
    criterion, log_path, epoch, numGroup=4, total_instance=4):
    # test_slides_info_list: [names, feats, labels]
    slide_names, slide_feats, slide_labels = test_slides_info_list
    distill = args.distill

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    instance_per_group = total_instance // numGroup

    gPred_0 = torch.FloatTensor().to(args.device)
    gt_0 = torch.LongTensor().to(args.device)
    gPred_1 = torch.FloatTensor().to(args.device)
    gt_1 = torch.LongTensor().to(args.device)

    total_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():

        numSlides = len(slide_names)
        numIter = numSlides // args.batch_size
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * args.batch_size:(idx + 1) * args.batch_size]
            tslide_names = [slide_names[sst] for sst in tidx_slide]
            tlabel = [slide_labels[sst] for sst in tidx_slide]
            batch_feat = [ slide_feats[sst].to(args.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslide_name = tslide_names[tidx]
                tslide_label = torch.from_numpy(tlabel[tidx]).unsqueeze(0).to(args.device)

                midFeat = dimReduction(tfeat)
                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
                allSlide_pred_softmax = []
                num_MeanInference = 1
                for jj in range(num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslide_label)
                        idx_tensor = torch.LongTensor(tindex).to(args.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0) # n
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict, bg_feat0, Att_s0 = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        # patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                        patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            # slide_d_feat.append(tattFeat_tensor)
                            slide_d_feat.append(bg_feat0)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    # test_loss0.update(loss0.item(), numGroup)

                    gSlidePred, slide_feat, Att_s1 = UClassifier(slide_d_feat)
                    # allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))
                    allSlide_pred_softmax.append(torch.sigmoid(gSlidePred)) # [1,1]

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslide_label], dim=0)

                # loss1 = F.nll_loss(allSlide_pred_softmax, tslide_label)
                loss1 = criterion(allSlide_pred_softmax, tslide_label)
                # test_loss1.update(loss1.item(), 1)

                total_loss = total_loss + loss0.item() + loss1.item()

                sys.stdout.write('\r Testing slide [%d/%d] slide loss: %.4f' % (idx, len(slide_names), loss0.item()))
                sys.stdout.write('\r Testing slide [%d/%d] slide loss: %.4f' % (idx, len(slide_names), loss1.item()))
                test_labels.extend(tslide_label.cpu().numpy())
                test_predictions.extend([allSlide_pred_softmax.squeeze().cpu().numpy()])

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction_slide = copy.deepcopy(test_predictions)
        class_prediction_slide[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_slide[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_slide
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
        
    else:        
        for i in range(args.num_classes):
            class_prediction_slide = copy.deepcopy(test_predictions[:, i])
            class_prediction_slide[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_slide[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_slide
            print(confusion_matrix(test_labels[:,i],test_predictions[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions[:,i])
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))

    slide_score = 0
    # average acc of all labels
    for i in range(0, len(slide_names)):
        slide_score = np.array_equal(test_labels[i], test_predictions[i]) + slide_score         
    avg_score = slide_score / len(slide_names)  #ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    print('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
    if epoch == args.num_epochs-1:
        log_rep = classification_report(test_labels, test_predictions, digits=4,output_dict=True)
        with open(log_path,'a+') as log_txt:
            log_txt.write('{:.2f},{:.2f},{:.2f},{:.2f} \n'.format(log_rep['macro avg']['precision']*100,log_rep['macro avg']['recall']*100,avg_score*100,sum(auc_value)/len(auc_value)*100))

    return total_loss / len(slide_names), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train IBMIL for DTFD')
    parser.add_argument('--seed', default=0, type=int, help='Seeds to reproduce experiments')
    parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU for training')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]') # binary classifier (positive and negative slides) -> num_classes 1
    parser.add_argument('--feature_extractor', default='resnet50-imagenet-transform', help='Name of feature extractor used')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size during training')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='slide classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and slide aggregating')
    
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    parser.add_argument('--weight_path', type=str, default=None, help='directory for loading pretrained model')
    parser.add_argument('--distill', type=str, default='MaxMinS', help='')
    
    args = parser.parse_args()
    assert args.model == 'DTFD'

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', str(args.dataset), str(args.feature_extractor), str(args.model)+'_'+str(args.distill), f"seed_{args.seed}")
    else:
        save_path = os.path.join('baseline', str(args.dataset), str(args.feature_extractor), str(args.model)+'_'+str(args.distill), f"seed_{args.seed}")
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    features_dir = Path("feature_extraction/outputs", args.dataset, args.feature_extractor, "slide")

    train_slides = {}
    test_slides = {}
    SUBSETS = ['train', 'test']

    if args.dataset == "camelyon16":
        CLASSES = ['normal', 'tumor']
    elif args.dataset == "tcga-nsclc":
        CLASSES = ['LUAD', 'LUSC']

    loss_weight = torch.tensor([[ 1]]).to(args.device)

    for subset in SUBSETS:
        for label, class_ in enumerate(sorted(CLASSES)):
            features_full_dir = f"{features_dir}/{subset}/{class_}"
            slide_features = [p for p in Path(features_full_dir).glob('*')]
            assert len(slide_features), f"There is no extracted slide features in {features_full_dir}"
            for slide_feature_path in slide_features:
                slide_feature_path = str(slide_feature_path)
                if subset == "train":
                    train_slides[slide_feature_path] = label
                elif subset == "test":
                    test_slides[slide_feature_path] = label

    train_slide_names, train_slide_feats, train_slide_labels = get_slides_info(train_slides, args.num_classes)
    test_slide_names, test_slide_feats, test_slide_labels = get_slides_info(test_slides, args.num_classes)
    print(f'Total training slides: {len(train_slide_names)}, Total testing slides: {len(test_slide_names)}')
    with open(log_path,'a+') as log_txt:
        log_txt.write(f'Total training slides: {len(train_slide_names)}, Total testing slides: {len(test_slide_names)}\n')

    '''
    model 
    1. choose model
    2. check the trainable params
    '''
    
    if args.model == 'DTFD':
        from Models.DTFD.network import DimReduction
        from Models.DTFD.Attention import Attention_Gated as Attention
        from Models.DTFD.Attention import Attention_with_Classifier, Classifier_1fc

        mDim = args.feats_size//2

        DTFDclassifier = Classifier_1fc(mDim, args.num_classes, 0.0).to(args.device)
        DTFDattention = Attention(mDim).to(args.device)
        DTFDdimReduction = DimReduction(args.feats_size, mDim, numLayer_Res=0).to(args.device)
        DTFDattCls = Attention_with_Classifier(args, L=mDim, num_cls=args.num_classes, \
            droprate=0.0, confounder_path=args.c_path).to(args.device)

        if args.weight_path:
            state_dict_weights = torch.load(args.weight_path) 
            msg = DTFDdimReduction.load_state_dict(state_dict_weights['dim_reduction'], strict=False)
            DTFDdimReduction.eval()
        milnet = [DTFDclassifier, DTFDattention, DTFDdimReduction, DTFDattCls]
    
    if not isinstance(milnet, list):
        milnet = []
    for sub_net in milnet:
        for name, _ in sub_net.named_parameters():
                print('Training {}'.format(name))
                with open(log_path,'a+') as log_txt:
                    log_txt.write('\n Training {}'.format(name))

    # loss, optim, schduler for DTFD
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight).to(args.device)
    trainable_parameters = []
    trainable_parameters += list(DTFDclassifier.parameters())
    trainable_parameters += list(DTFDattention.parameters())
    if args.weight_path:
        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=1e-4,  weight_decay=args.weight_decay)
        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, [int(args.num_epochs/2)], gamma=0.2)
    else:
        trainable_parameters += list(DTFDdimReduction.parameters())
        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=1e-4,  weight_decay=args.weight_decay)
        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, [int(args.num_epochs/2)], gamma=0.2)
    optimizer_adam1 = torch.optim.Adam(DTFDattCls.parameters(), lr=1e-4,  weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, [int(args.num_epochs/2)], gamma=0.2)

    best_score = 0

    train_slides_info_list = (train_slide_names, train_slide_feats, train_slide_labels)
    test_slides_info_list = (test_slide_names, test_slide_feats, test_slide_labels)

    for epoch in range(1, args.num_epochs+1):
        start_time = time.time()

        train_loss_slide = trainDTFD(args, train_slides_info_list, DTFDclassifier, \
            DTFDdimReduction, DTFDattention, DTFDattCls, optimizer_adam0, optimizer_adam1, epoch, criterion,\
                log_path=log_path)
        print('epoch time:{}'.format(time.time()- start_time))
        #test_loss_slide, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        test_loss_slide, avg_score, aucs, thresholds_optimal = \
            testDTFD(args, test_slides_info_list, DTFDclassifier, DTFDdimReduction, DTFDattention, \
                DTFDattCls, criterion, log_path, epoch)
        
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_slide, test_loss_slide, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
        with open(log_path,'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_slide, test_loss_slide, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        if scheduler0:
            scheduler0.step()
        scheduler1.step()
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            tsave_dict = {
                'classifier': DTFDclassifier.state_dict(),
                'dim_reduction': DTFDdimReduction.state_dict(),
                'attention': DTFDattention.state_dict(),
                'att_classifier': DTFDattCls.state_dict()
            }
            torch.save(tsave_dict, save_name)

            with open(log_path,'a+') as log_txt:
                info = 'Best model saved at: ' + save_name +'\n'
                log_txt.write(info)
                info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                log_txt.write(info)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    log_txt.close()

if __name__ == '__main__':
    main()