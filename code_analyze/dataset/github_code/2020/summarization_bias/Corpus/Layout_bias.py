
# coding: utf-8

# In[2]:


import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


path_ref = ''
path_doc = ''


# In[4]:


def line_plot(doc_importance):
    index = np.arange(doc_importance.shape[0])
    label = list(np.arange(1,doc_importance.shape[0]+1))
    importance = list(doc_importance)
    plt.plot(index,importance, linestyle='solid')
    plt.xlabel('Sentence number', fontsize=5)
    plt.ylabel('Imprtance Value', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Importance throughout document')
    plt.show()


# In[33]:


def bar_plot(doc_importance):
    index = np.arange(doc_importance.shape[0])
    label = list(np.arange(1,doc_importance.shape[0]+1))
    importance = list(doc_importance)
    plt.bar(index, importance)
    plt.xlabel('Sentence number', fontsize=5)
    plt.ylabel('Imprtance Value', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Importance throughout document')
    plt.show()


# In[28]:


def cos_sim(topic,ground_truth):
    topic_path = path_doc + '/' + topic
    documents = os.listdir(topic_path)
    reference_path = path_ref + '/' + ground_truth + '/' + topic + ground_truth + '.txt'
    ref_matrix = np.loadtxt(reference_path, dtype=float)
    for d in documents:
        document_path = topic_path + '/' + d
        doc_matrix = np.loadtxt(document_path,dtype=float)
#         print(doc_matrix.shape)
#         print(ref_matrix.shape)
        if doc_matrix.ndim<2 or ref_matrix.ndim<2:
            continue
        sim_val = cosine_similarity(doc_matrix,ref_matrix)
        doc_importance = np.amax(sim_val,axis=1)
        doc_importance = doc_importance.flatten()
        sentences = doc_importance.shape[0]
        if sentences<3:
            continue
        sent_per_part = int(sentences/3)
#         print(sent_per_part)
        remaining = sentences%sent_per_part
        rows = (sentences-remaining)/sent_per_part
        temporary_importance = doc_importance[0:int(doc_importance.shape[0]-remaining)].reshape(int(rows),sent_per_part)
        averages = np.mean(temporary_importance, axis=1)
        if(remaining>0):
            averages = list(averages)
            averages.append(np.mean(doc_importance[int(doc_importance.shape[0]-remaining):doc_importance.shape[0]]))
            averages = np.asarray(averages)
        if len(averages)>3:
            sum_ = sum(averages[2:len(averages)])
            averages[2] = sum_/(len(averages)-2)
        averages = averages[0:3]
#         print(len(averages))
        first_prt.append(averages[0])
        second_prt.append(averages[1])
        third_prt.append(averages[2])
#         print(doc_importance.shape)
#         bar_plot(doc_importance)
#         line_plot(averages)


# In[12]:


topics = os.listdir(path_doc)


# In[27]:


cos_sim('D30040','A')


# In[29]:


first_prt = []
second_prt = []
third_prt = []
for t in topics:
    print(t)
    cos_sim(t,'A')


# In[30]:


print(sum(first_prt)/len(first_prt))
print(sum(second_prt)/len(second_prt))
print(sum(third_prt)/len(third_prt))

