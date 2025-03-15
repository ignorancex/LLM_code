from transformers import * 
import os 
import torch 
import json
import numpy as np 

from model import MemeDialoGPT 
from dataset import MODDataset, get_data 
from utils import accuracy_compute, AverageMeter, meme_classify_accuracy
from torch.utils.data import DataLoader 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

# data parameters
train_data_path = 'data/dialog/toy_data.json'
#train_data_path = 'data/dialog/toy_data.json' 
val_data_path = 'data/dialog/toy_data.json' 
feature_path = 'data/meme/id2feature.json'
#feature_path = 'data/meme/id2feature.json'


# model parameters
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
model_path = 'ckpt/mod_gpt' 
gpt_path = 'ckpt/origin_gpt'
ckpt_usage = False 
lr = 6e-5
epochs = 10 
gradient_accumulation_steps = 1
print_freq = 1


def main(): 
    
    # model initialize 
    if ckpt_usage == True: 
        ckpt_path = 'ckpt/mod_gpt/model.bin' 
         

    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_path, do_lower_case=True)
        model = MemeDialoGPT.from_pretrained(gpt_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr) 

    # data read 
    train_dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path) 
    val_dialogs, _ = get_data(tokenizer, val_data_path, feature_path) 
    #print(len(train_dialogs))
    train_dataset = MODDataset(train_dialogs, id2feature, tokenizer) 
    val_dataset = MODDataset(val_dialogs, id2feature, tokenizer) 
    # print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True) 
    
    for epoch in range(epochs): 
        
        # one epoch's training
        val_loss = train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_loader, epoch=epoch) 
        
        # one epoch's validation 
        validate(model=model, tokenizer=tokenizer, dataset=val_loader, epoch=epoch)
        
        break
        # save checkpoint 
        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
            '%s/epoch_%d_loss_%.3f'%(model_path, epoch, val_loss))
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        tokenizer.save_vocabulary(model_path)


def train(model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    cat_img_features = img_feature_read(feature_path) 
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    iteration = 1
    meme_correct_num = 0 
    meme_total_num = 0

    for instance in dataset: 
        history_txt, history_img, token_type_ids, labels, meme_flag = instance 
        history_txt, history_img, token_type_ids, labels, meme_flag = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), \
                                                                        token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(0), meme_flag.to(device).squeeze(0)   
        history_txt_embs = model.transformer.wte(history_txt) 
        #print(history_txt_embs.size()) 
        history_img_embs = model.img_ff(history_img) 
        #print(history_img_embs.size()) 
        #print(token_type_ids) 
        #print(history_txt)
        input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
        input_embs = input_embs.to(device) 
        img_feature = history_img[-1, :].unsqueeze(0)
        # print(input_embs.size()) 
        # print(img_feature.size()) 
        loss, mf_logits, lm_logits, cur_img_feature = model(input_embs, token_type_ids, labels, img_feature, meme_flag) 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if img_feature[0][0] != 0.: 
            if meme_retrieval_compute(cur_img_feature, img_feature, cat_img_features):
                meme_correct_num += 1 
            meme_total_num += 1 
        # acc = accuracy_compute(lm_logits, labels, 5)
        acc = meme_classify_accuracy(mf_logits, meme_flag).item()
        avg_acc.update(acc)
        
        avg_loss.update(loss.item())
        
        # print status 
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Classify Acc {acc.val:.3f} ({acc.avg:.3f})\t'
            'Meme Acc {mac:.3f}'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc, mac=float(meme_correct_num/meme_total_num)))
        
        iteration += 1 
        break
    return avg_loss.avg
        # print(loss)
        # break  


# concatenate the input 
def input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer): 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    emb_length = token_type_ids.size(-1) 
    emb_dim = history_txt_embs.size(-1)
    img_num = history_img_embs.size(0) 

    input_embs = torch.zeros((emb_length, emb_dim)).to(device)

    txt_idx = 0 
    img_idx = 0 
    left_idx  = 0 
    right_idx = 0 
    while right_idx < emb_length: 
        #if right_idx == emb_length-1 and token_type_ids[right_idx] == img: 
        #    break 
        if right_idx < emb_length-1 and token_type_ids[right_idx] == img:
            txt_length = right_idx - left_idx 
            input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:txt_idx+txt_length, :] 
            txt_idx += txt_length 
            input_embs[right_idx,:] = history_img_embs[img_idx, :] 
            img_idx += 1
            left_idx = right_idx + 1 
        right_idx += 1
    txt_length = right_idx - left_idx 
    if txt_length > 0: 
        input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:, :]
    # img_feature = history_img_embs[img_idx,:] 
    return input_embs



def validate(model, tokenizer, dataset, epoch): 
    
    model.eval() 
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    avg_bleu = AverageMeter() 
    iteration = 1 
    cat_img_features = img_feature_read(feature_path) 
    meme_correct_num = 0 
    meme_total_num = 0

    with torch.no_grad(): 
        for instance in dataset: 
            history_txt, history_img, token_type_ids, labels, meme_flag = instance 
            history_txt, history_img, token_type_ids, labels, meme_flag  = history_txt.to(device).squeeze(0), history_img.to(device).squeeze(0), \
                                                                            token_type_ids.to(device).squeeze(0), labels.to(device).squeeze(0), meme_flag.to(device).squeeze(0) 
            history_txt_embs = model.transformer.wte(history_txt) 
            history_img_embs = model.img_ff(history_img) 
            
            input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
            input_embs = input_embs.to(device) 
            if input_embs.size(-2) > 450:
                continue
            img_feature = history_img[-1, :].unsqueeze(0) 
            loss, mf_logits, lm_logits, cur_img_feature = model(input_embs, token_type_ids, labels, img_feature, meme_flag) 
            # compare cur_img_feature is among topk with img_feature 
            # print(cur_img_feature.size())   (1, 512) 
            if img_feature[0][0] != 0.: 
                if meme_retrieval_compute(cur_img_feature, img_feature, cat_img_features):
                    meme_correct_num += 1 
                meme_total_num += 1 
            #acc = accuracy_compute(lm_logits, labels, k=5) 
            acc = meme_classify_accuracy(mf_logits, meme_flag).item()
            avg_acc.update(acc) 
            avg_loss.update(loss.item()) 
            if iteration % print_freq == 0:
                print('Epoch:[{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                'Meme Acc {mac:.3f}'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc, mac=float(meme_correct_num/meme_total_num))) 
            iteration += 1 
            break 



def img_feature_read(feature_path): 
    with open(feature_path, 'r', encoding='utf-8') as f: 
        id2feature_dict = json.load(f) 
    img_features = [] 
    for id in id2feature_dict.keys():
        img_features.append(id2feature_dict[id]) 
    img_features = np.array(img_features) 
    img_features = torch.from_numpy(img_features).float().to(device) 
    return img_features 


def meme_retrieval_compute(cur_img_feature, target_img_feature, cat_img_features): 
    # (1, 512)
    cur_dist = torch.dist(cur_img_feature, target_img_feature, p=2)
    # print(cat_img_features.size())
    cur_img_list = cur_img_feature.repeat(307,1) 
    total_dist = torch.sqrt(torch.sum((cur_img_list - cat_img_features)**2, dim=1))
    # print(total_dist) 
    sorted_total, _ = torch.sort(total_dist) 
    # print(sorted_total) 
    return torch.gt(sorted_total[90],cur_dist)



if __name__ == '__main__': 
    main()
