import torch 

def collate_mismatch(batch):

    labels = [item[0] for item in batch]
    # print(labels)
    imgs = [item[1] for item in batch] 
    labels = torch.stack(labels,dim=0)
    imgs = torch.stack(imgs, dim=0)
    captions_batch = [item[2] for item in batch]
    # print("Captions Batch:", captions_batch) 
    captions_batch = torch.cat(captions_batch, dim=0)

    
    domain = [item[4] for item in batch]


    return labels, imgs, captions_batch,domain