#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pl.read_csv('dataset/train.csv', infer_schema_length=10000)
df.head()


# In[3]:


for i in range(13):
    col = f'I{i+1}'
    print(f'{col}: {df[col].min()} - {df[col].max()}')
    
    
for c in range(26):
    col = f'C{c+1}'
    print(f'{col}: {len(df[col].unique())}')


# In[4]:


from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder


# In[5]:


from numba import njit


class ColumnCatEncoder:
    
    def __init__(self, name):
        self.col = name
    
    def __call__(self, df):
        if not hasattr(self, 'map_dict'):
            self.map_dict = {}
            vals = df[self.col].value_counts().sort('counts')
            for i, (v, _) in enumerate(vals.iter_rows()):
                self.map_dict[v] = i
            if None not in self.map_dict:
                self.map_dict[None] = i
        n_df = df.with_columns(pl.col(self.col).\
                               map_dict(self.map_dict, \
                                        default=self.map_dict[None]))
        return n_df
    
    
class ColumnCatEncoderChance:
    
    def __init__(self, name):
        self.col = name
    
    def __call__(self, df):
        if not hasattr(self, 'map_click'):
            self.map_click = {}
            vals = df[[self.col, 'label']].\
                    groupby(self.col).\
                    agg([pl.mean('label'), pl.count()]).\
                    sort('count')
            for v, click, _ in vals.iter_rows():
                self.map_click[v] = click
            if None not in self.map_click:
                self.map_click[None] = self.map_click[v]
        df = df.with_columns(pl.col(self.col).\
                             map_dict(self.map_click, default=self.map_click[None]).\
                             alias(f'I{self.col}_label'))
        df = df.drop(self.col)
        return df
## TODO FAST coso        

class LogBaseEncoder:
    
    def __init__(self, col):
        self.col = col
        
    def __call__(self, df):
        col = df[self.col].fill_null(strategy='mean').to_numpy()
        if not hasattr(self, 'mean'):
            self.mean = np.mean(col)
            self.std = np.std(col)
            # print(self.mean)
            # print(self.std)
        res = (col-self.mean) / (3 * self.std)
        res = np.where(np.abs(res) <= 1, res, np.sign(res) * (np.log2(np.abs(res)) + 1))
        df = df.with_columns(pl.Series(name=self.col, values=res))
        return df


steps = [FunctionTransformer(ColumnCatEncoder(f'C{c}') 
                             if len(df[f'C{c}'].value_counts()) > 20 
                             else ColumnCatEncoderChance(f'C{c}')) 
         for c in range(1, 27)]

#Data already between 0-1
#steps += [FunctionTransformer(LogBaseEncoder(f'I{c}'))
#         for c in range(1, 14)]


    
pipe = make_pipeline(*steps)


# In[6]:


#get_ipython().run_cell_magic('time', '', 'pipe.fit(df)\n')
pipe.fit(df)


# In[7]:


#get_ipython().run_cell_magic('time', '', 'train = pipe.transform(df)\n')
train = pipe.transform(df)
del df

# In[8]:


train.head()


# In[12]:


test = pl.read_csv('dataset/test.csv', infer_schema_length=10000)


# In[13]:


test.head()


# In[14]:


test = pipe.transform(test)


# In[15]:

valid = pl.read_csv('dataset/valid.csv', infer_schema_length=10000)

valid = pipe.transform(valid)


# # Model and prediction

# In[19]:


import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from tqdm.auto import tqdm
import polars as pl
import numpy as np
import random
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# In[17]:


class CriteoDataset(Dataset):
    
    def __init__(self, df, cat=None, cat_max_val=None, num=None):
        if cat is None:
            self.cat = [c for c in df.columns if c.startswith('C')]
            self.cat_max_val = {x: df[x].max() + 1 for x in self.cat}
            self.num = [c for c in df.columns if c.startswith('I')]
        else:
            self.cat = cat
            self.cat_max_val = cat_max_val
            self.num = num
        self.cat_values = []
        for c in self.cat:
            self.cat_values.append(df[c].to_numpy().copy())#[:, np.newaxis])
        self.num_values = df[self.num].to_numpy().astype(np.float32)
        self.y = df[['label']].to_numpy()
    
    def __getitem__(self, idx):
        x = tuple(c[idx] for c in self.cat_values) + (self.num_values[idx, :],)
        return x, self.y[idx, :]
                                                      
    def __len__(self):
        return self.num_values.shape[0]


# In[23]:


ds_train = CriteoDataset(train)
ds_test = CriteoDataset(test)
ds_valid = CriteoDataset(valid)

del train
del test
del valid

# In[41]:


class EmbeddedFeatures(torch.nn.Module):
    
    def __init__(self, ds_train, dims=32):
        super().__init__()
        embs = []
        for c in ds_train.cat:
            m = ds_train.cat_max_val[c]
            e = torch.nn.Embedding(m, dims)
            embs.append(e)
        self.embeddings = torch.nn.ModuleList(embs)
        
    def forward(self, cats):
        embs = None
        for c, e in zip(cats, self.embeddings):
            if embs is None:
                embs = e(c)
            else:
                embs += e(c)
        embs /= len(cats)
        return embs

class DeepFeatures(torch.nn.Module):
    
    def __init__(self, ds_train, embs, depth=3, dims=32):
        super().__init__()
        self.embs = embs
        num_dims = ds_train.num_values.shape[1]
        #first 
        deep_list = [torch.nn.Linear(dims + num_dims, dims)]
        for _ in range(1, depth):
            deep_list.append(torch.nn.Linear(dims, dims))
        self.deep = torch.nn.ModuleList(deep_list)
        
        
    def forward(self, cats, nums, std=0.10):
        embs = self.embs(cats)
        x = torch.cat((embs, nums), dim=1)
        if self.training:
            x = x * (1 + std * torch.randn_like(x))
        output = []
        for l in self.deep:
            x = l(x)
            if self.training:
                x = x * (1 + std * torch.randn_like(x))
            output.append(x)
            x = torch.nn.functional.leaky_relu(x)
        return output
    

class DeepMF(torch.nn.Module):
    
    def __init__(self, ds_train, depth=3, dims=32):
        super().__init__()
        embds = EmbeddedFeatures(ds_train, dims=dims)
        self.base = DeepFeatures(ds_train, embds, depth=depth, dims=dims)
        self.click = DeepFeatures(ds_train, embds, depth=depth, dims=dims)
        self.multi = torch.nn.parameter.Parameter(torch.randn((1,1)))
        self.att = torch.nn.parameter.Parameter(torch.randn((depth, 1)))
        
        
    def forward(self, cats, nums):
        base = self.base(cats, nums)
        click = self.click(cats, nums)
        click_out = None
        for e, (b, c) in enumerate(zip(base, click)):
            c_v = torch.sum(b * c, dim=1, keepdim=True) * self.multi
            if click_out is None:
                click_out = c_v * self.att[e, 0]
            else:
                click_out += (c_v * self.att[e, 0])
        out = click_out
        return torch.nn.functional.sigmoid(out)


# In[42]:


device = 'cuda'


# In[43]:


model = DeepMF(ds_train).to(device)


# In[57]:


dl_train = DataLoader(ds_train, batch_size=1024, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=30000, shuffle=False)
dl_valid = DataLoader(ds_valid, batch_size=30000, shuffle=False)


# In[45]:


def epoch(model, loss_f, optimizer, dl_train, device=device):
    loss = 0
    for x, y in tqdm(dl_train):
        optimizer.zero_grad()
        cats = [c.to(device) for c in x[:-1]]
        nums = x[-1].to(device)
        y = y.float().to(device)
        y_pred = model(cats, nums)
        c_loss = loss_f(y_pred, y)
        c_loss.backward()
        optimizer.step()
        loss += c_loss.cpu().item()
    return loss / len(dl_train)


# In[52]:


def predict(model, dl_test, device=device):
    preds = [] 
    real = []
    with torch.no_grad():
        for x, y in tqdm(dl_test):
            cats = [c.to(device) for c in x[:-1]]
            nums = x[-1].to(device)
            y_pred = model(cats, nums).cpu().numpy()
            preds.append(y_pred)
            real.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    real  = np.concatenate(real, axis=0)
    return real, preds


# In[47]:


loss_f = torch.nn.BCELoss()
optimizer = torch.optim.RAdam(model.parameters())


# In[56]:


from sklearn.metrics import roc_auc_score
if not os.path.exists('criteo-10-10n.pt'):
    best = float('inf')
    for i in range(10):
        model.train()
        l = epoch(model, loss_f, optimizer, dl_train)
        print(f'{i}: Current loss in training {l}')
        model.eval()
        y, pred = predict(model, dl_valid)
        logloss = loss_f(torch.from_numpy(pred), torch.from_numpy(y.astype(np.float32))).item()
        print(f'Valid AUC Score: {roc_auc_score(y, pred)}')
        print(f'Valid LogLoss: {logloss}')
        if logloss < best:
            print('Improved!')
            best = logloss
            torch.save(model.state_dict(), 'criteo-10-10n-best.pt')
        y, pred = predict(model, dl_test)
        print(f'Test AUC Score: {roc_auc_score(y, pred)}')
        print(f'Test LogLoss: {loss_f(torch.from_numpy(pred), torch.from_numpy(y.astype(np.float32))).item()}')
    torch.save(model.state_dict(), 'criteo-10-10n.pt')
else:
    model.load_state_dict(torch.load('criteo-10-10n.pt'))


# In[58]:


model.eval()
y, pred = predict(model, dl_test)

print(f'AUC Score: {roc_auc_score(y, pred)}')
print(f'LogLoss: {loss_f(torch.from_numpy(pred), torch.from_numpy(y.astype(np.float32))).item()}')

# In[ ]:




