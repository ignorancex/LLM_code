#Dataloader 
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import clip 

data_100= json.load(open('./nc_all_data_train.json'))
data_75 = json.load(open('./nc_all_domain_75_percent_data.json'))
data_50 = json.load(open('./nc_all_domain_50_percent_data.json'))
data_25 = json.load(open('./nc_all_domain_25_percent_data.json'))
data_5 = json.load(open('./nc_all_domain_5_percent_data.json'))
test_data=json.load(open('./nc_all_data_test.json'))
dom100 = list(set( [data_100[idx]['topic'] for idx in data_100]))
dom75 = list(set( [data_75[idx]['topic'] for idx in data_75]))
dom50 = list(set( [data_50[idx]['topic'] for idx in data_50]))
dom25 = list(set( [data_25[idx]['topic'] for idx in data_25]))
dom5= list(set( [data_5[idx]['topic'] for idx in data_5]))
test_data= list(set( [test_data[idx]['topic'] for idx in test_data]))

def find_intersection(lists):
    if not lists:
        return []
    
    result = set(lists[0])
    
    for lst in lists[1:]:
        result.intersection_update(lst)
    
    return list(result)


common_domains = find_intersection([dom100, dom75, dom50, dom25, dom5,test_data])


def ohe(arr):
    

    ohe_dict = {}
    categories = arr
    c=0
    
    # Create a dictionary to map categories to unique integers
    category_to_int = {category: i for i, category in enumerate(np.unique(categories))}

    # Convert the original categorical data to integer codes
    int_codes = np.array([category_to_int[category] for category in categories])

    # Determine the number of unique categories
    num_categories = len(category_to_int)

    # Create an empty one-hot encoded matrix
    one_hot_encoded = np.zeros((len(categories), num_categories))

    # Fill the one-hot encoded matrix
    one_hot_encoded[np.arange(len(categories)), int_codes] = 1
    
    for k in arr:
        ohe_dict[k] = one_hot_encoded[c]
        c+=1
    
    return ohe_dict
class NewsClip(Dataset):
    def __init__(self, data_paths_nc, visual_news_root_dir, news_clip_root_dir, split, transform):

        self.data_paths_nc = data_paths_nc

        self.transform = transform

        self.domains = ['world', 'travel', 'arts_culture', 'disaster_accident', 'sports',
       'news', 'business_economy', 'books', 'theobserver', 'artanddesign',
       'australia-news', 'cities', 'law_crime', 'environment',
       'conflict_attack', 'theguardian', 'media', 'business', 'sport',
       'commentisfree', 'sustainable-business', 'global-development',
       'culture', 'healthcare-network',
       'global-development-professionals-network',
       'voluntary-sector-network', 'housing-network',
       'local-government-network', 'society', 'science_technology',
       'public-leaders-network', 'fashion', 'tv-and-radio',
       'international_relations', 'leeds', 'music', 'politics_elections',
       'lifeandstyle', 'childrens-books-site', 'religion', 'film',
       'uk-news', 'football', 'science', 'uk', 'politics', 'us-news',
       'money', 'technology', 'education', 'weather', 'stage', 'law',
       'health_medicine_environment']

        self.ohe_dom = ohe(np.array(self.domains))
        

        
    def __len__(self):
        return len(self.data_paths_nc)   

    def load_img_pil(self,image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
                       
    def load_queries(self,visual_news_caption_item, visual_news_image_item):
        # caption = visual_news_caption_item['caption']
        caption =  str(visual_news_caption_item) 
        # print("Caption is: ", caption)
        image_path = visual_news_image_item
        # caption_tokenized = caption
        caption_tokenized = clip.tokenize(caption,truncate=True)
        caption_unfiltered = caption 
        # image_path = os.path.join(self.visual_news_root_dir, visual_news_image_item['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        
        return transform_img, caption_tokenized, caption_unfiltered



    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        # print(idx)
        label = torch.as_tensor(0) if self.data_paths_nc[str(idx)]['real'] else torch.as_tensor(1)
        
        # print(label.item())
        visual_news_caption_item = self.data_paths_nc[str(idx)]["caption"]
        visual_news_image_item = self.data_paths_nc[str(idx)]["image_path"]

        qImg, qCap, caption_unfiltered = self.load_queries(visual_news_caption_item, visual_news_image_item)
        domain = self.data_paths_nc[str(idx)]['dom_embed']   
        # domain = self.ohe_dom[self.data_paths_nc[str(idx)]["topic"]] # If you want to use One hot vector encoding then use this
 

        return label, qImg, qCap,  caption_unfiltered, domain




        

