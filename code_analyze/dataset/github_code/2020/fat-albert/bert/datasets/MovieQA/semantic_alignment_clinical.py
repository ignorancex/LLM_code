import config as cfg
import story_loader
import data_loader
import csv
import numpy as np
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
num_sentences = 10
total_sentences = 5
mqa = data_loader.DataLoader()

def common_member(a, b):
    common_set = []
    counter = 0
    for i in a:
        for j in b:
            if i == j and counter < total_sentences:
                common_set.append(i)
                counter = counter + 1 
    return common_set  

def stringReplace(input):
    output = input
    if not output:
        output = 'None'
    return output

## LOAD THE TEST DATASET
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'split_plot')
with open('clinical.csv', mode='w') as test_file:
    employee_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    counter = 0
    for q in qa:
        movie_idx = q[4]
        movie_story = ', '.join(story[movie_idx])
        question = q[1]
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        index = 0
        #story_split = movie_story.splitlines()
        movie_align = ''
        sen_mtx = []
        comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
        sim_mtx_w = []
        sim_mtx_c = []
        for sen in story[movie_idx]:
            sen_mtx.append((sen,comparison_metric))
        sim_mtx_w = web_model.predict(sen_mtx)
        sim_mtx_c = clinical_model.predict(sen_mtx)
        #sim_mtx = sim_mtx_w + sim_mtx_c
        #print(sim_mtx_w)
        #print(sim_mtx_c)
        sim_mtx = sim_mtx_w
        #sim_mtx.append(sim_mtx_c)
        np.concatenate((sim_mtx_w, sim_mtx_c), axis=None)
        print(len(sen_mtx))
        print("-----------")
        top_sentences_w = np.asarray(sim_mtx_w).argsort()[-num_sentences:][::-1]
        top_sentences_c = np.asarray(sim_mtx_c).argsort()[-num_sentences:][::-1]
        top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]

        #top_sentences = common_member(top_sentences_w, top_sentences_c)
        print(counter,q[5],top_sentences)
        counter = counter + 1
