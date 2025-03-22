import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys, argparse, os, copy, glob
import numpy as np
import time
import random
import json

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from pathlib import Path

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

def train(train_slides_info_list, milnet, criterion, optimizer, args, log_path, epoch=0):
    # train_slides_info_list: [names, feats, labels]
    slide_names, slide_feats, slide_labels = train_slides_info_list
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0

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

            optimizer.zero_grad()

            if args.model == 'dsmil':
                ins_prediction, bag_prediction, attention, atten_B= milnet(tslide_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)     
                bag_loss = criterion(bag_prediction.view(1, -1), tslide_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), tslide_label.view(1, -1))
                loss = 0.5*bag_loss + 0.5*max_loss

            elif args.model =='abmil':
                bag_prediction, _, attention = milnet(tslide_feats)
                loss =  criterion(bag_prediction.view(1, -1), tslide_label.view(1, -1))

            
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            atten_max = atten_max + attention.max().item()
            atten_min = atten_min + attention.min().item()
            atten_mean = atten_mean +  attention.mean().item()
            
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f, attention max:%.4f, min:%.4f, mean:%.4f' % (idx, len(slide_names), loss.item(), 
                            attention.max().item(), attention.min().item(), attention.mean().item()))
        
    atten_max = atten_max / len(slide_names)
    atten_min = atten_min / len(slide_names)
    atten_mean = atten_mean / len(slide_names)
    
    with open(log_path,'a+') as log_txt:
            log_txt.write('\n atten_max'+str(atten_max))
            log_txt.write('\n atten_min'+str(atten_min))
            log_txt.write('\n atten_mean'+str(atten_mean))

    return total_loss / len(slide_names)


def test(test_slides_info_list, milnet, criterion, optimizer, args, log_path, epoch):
    # test_slides_info_list: [names, feats, labels]
    slide_names, slide_feats, slide_labels = test_slides_info_list
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():

        numSlides = len(slide_names)
        numIter = numSlides // args.batch_size
        tIDX = list(range(numSlides)) # not random

        for idx in range(numIter):
            
            tidx_slide = tIDX[idx * args.batch_size:(idx + 1) * args.batch_size]
            tslide_names = [slide_names[sst] for sst in tidx_slide]
            tlabel = [slide_labels[sst] for sst in tidx_slide]
            batch_feat = [ slide_feats[sst].to(args.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslide_name = tslide_names[tidx]
                tslide_label = torch.from_numpy(tlabel[tidx]).unsqueeze(0).to(args.device)

                if args.model == 'dsmil':
                    ins_prediction, bag_prediction, _, _ = milnet(tfeat)
                    max_prediction, _ = torch.max(ins_prediction, 0)  
                    bag_loss = criterion(bag_prediction.view(1, -1), tslide_label.view(1, -1))
                    max_loss = criterion(max_prediction.view(1, -1), tslide_label.view(1, -1))
                    loss = 0.5*bag_loss + 0.5*max_loss

                elif args.model in ['abmil', 'max', 'mean']:
                    bag_prediction, _, _ =  milnet(tfeat)
                    max_prediction = bag_prediction
                    loss = criterion(bag_prediction.view(1, -1), tslide_label.view(1, -1))

                total_loss = total_loss + loss.item()
                sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (idx, len(slide_names), loss.item()))
                test_labels.extend(tslide_label.cpu().numpy())
                if args.average:   # notice args.average here
                    test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
                else: 
                    test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print('\n')
        print(confusion_matrix(test_labels,test_predictions))
        info = confusion_matrix(test_labels,test_predictions)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:,i],test_predictions[:,i]))
            info = confusion_matrix(test_labels[:,i],test_predictions[:,i])
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n'+str(info))
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(slide_names)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(slide_names)  #ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)
    print('\n  dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n dsmil-metrics: multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
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
    parser = argparse.ArgumentParser(description='Train IBMIL for abmil and dsmil')
    parser.add_argument('--seed', default=0, type=int, help='Seeds to reproduce experiments')
    parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU for training')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]') # binary classifier (positive and negative bags) -> num_classes 1
    parser.add_argument('--feature_extractor', default='resnet50-imagenet-transform', help='Name of feature extractor used')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size during training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')

    parser.add_argument('--agg', type=str,help='which agg')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    # for ablations only
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')
    parser.add_argument('--c_dim', default=128, type=int, help='Dimension of the projected confounders')
    parser.add_argument('--freeze_epoch', default=999, type=int, help='freeze confounders during this many epoch from the start')
    parser.add_argument('--c_merge', type=str, default='cat', help='cat or add or sub')

    args = parser.parse_args()

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', str(args.dataset), str(args.feature_extractor), str(args.model), f"seed_{args.seed}")
    else:
        save_path = os.path.join('baseline', str(args.dataset), str(args.feature_extractor), str(args.model), f"seed_{args.seed}")
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

    if args.model == 'dsmil':
        import dsmil as mil
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(args.device)
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,confounder_path=args.c_path).to(args.device)
        milnet = mil.MILNet(i_classifier, b_classifier).to(args.device)

    elif args.model == 'abmil':
        import abmil as mil
        milnet = mil.Attention(in_size=args.feats_size, out_size=args.num_classes,confounder_path=args.c_path, \
            confounder_learn=args.c_learn, confounder_dim=args.c_dim, confounder_merge=args.c_merge).to(args.device)
    
    for name, _ in milnet.named_parameters():
        print('Training {}'.format(name))
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n Training {}'.format(name))

    # sanity check begins here
    print('*******sanity check *********')
    for k,v in milnet.named_parameters():
        if v.requires_grad == True:
            print(k)

    # loss, optim, schduler
    criterion = nn.BCEWithLogitsLoss() 
    original_params = []
    confounder_parms = []
    for pname, p in milnet.named_parameters():
        if ('confounder' in pname):
            confounder_parms += [p]
            print('confounders:',pname )
        else:
            original_params += [p]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), 
                                lr=args.lr, betas=(0.5, 0.9), 
                                weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    best_score = 0 

    train_slides_info_list = (train_slide_names, train_slide_feats, train_slide_labels)
    test_slides_info_list = (test_slide_names, test_slide_feats, test_slide_labels)
    
    
    for epoch in range(1, args.num_epochs+1):
        start_time = time.time()

        train_loss_bag = train(train_slides_info_list, milnet, criterion, optimizer, args, log_path, epoch=epoch-1) # iterate all bags
        print('epoch time:{}'.format(time.time()- start_time))
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_slides_info_list, milnet, criterion, optimizer, args, log_path, epoch)
        
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
        with open(log_path,'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

        scheduler.step()
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            with open(log_path,'a+') as log_txt:
                info = 'Best model saved at: ' + save_name +'\n'
                log_txt.write(info)
                info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                log_txt.write(info)
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
        if epoch == args.num_epochs:
            save_name = os.path.join(save_path, 'last.pth')
            torch.save(milnet.state_dict(), save_name)
    log_txt.close()

if __name__ == '__main__':
    main()
