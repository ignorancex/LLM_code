import config as cfg
import story_loader
import data_loader
import csv
import numpy as np
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cuda', batch_size=100) #defaults to GPU prediction
clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=100) #defaults to GPU prediction

num_sentences = 6
mqa = data_loader.DataLoader()

def stringReplace(input):
    output = input
    if not output:
        output = 'None'
    return output

## LOAD THE VAL DATASET and WEB
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'split_plot')
with open('sem_align_eval/val_com.csv', mode='w') as out_file:
    file_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['', 'plot_align' , 'similarity_align'])
    counter = 0
    for q in qa:
        movie_idx = q[4]
        question = q[1]
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        index = 0
        movie_align = ''
        sim_mtx = []
        comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
        sen_mtx = []
        for sen in story[movie_idx]:
            sen_mtx.append((sen,comparison_metric))
        sim_mtx_w = web_model.predict(sen_mtx)
        sim_mtx_c = clinical_model.predict(sen_mtx)
        sim_mtx = np.divide(sim_mtx_w,np.mean(sim_mtx_w)) + np.divide(sim_mtx_c,np.mean(sim_mtx_c))
        top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]
        for a in top_sentences:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n"))
        print("COMB",counter)
        file_writer.writerow([counter, q[5], top_sentences])
        correct_ans = q[3]
        counter = counter + 1

## LOAD THE VAL DATASET and CLINICAL
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'split_plot')
with open('sem_align_eval/val_clin.csv', mode='w') as out_file:
    file_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['', 'plot_align' , 'similarity_align'])
    counter = 0
    for q in qa:
        movie_idx = q[4]
        question = q[1]
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        index = 0
        movie_align = ''
        sim_mtx = []
        comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
        sen_mtx = []
        for sen in story[movie_idx]:
            sen_mtx.append((sen,comparison_metric))
        #sim_mtx_w = web_model.predict(sen_mtx)
        sim_mtx = clinical_model.predict(sen_mtx)
        #sim_mtx = np.divide(sim_mtx_w,np.mean(sim_mtx_w)) + np.divide(sim_mtx_c,np.mean(sim_mtx_c))
        top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]
        for a in top_sentences:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n"))
        print("CLIN",counter)
        file_writer.writerow([counter, q[5], top_sentences])
        correct_ans = q[3]
        counter = counter + 1


## LOAD THE VAL DATASET and WEB
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'split_plot')
with open('sem_align_eval/val_web.csv', mode='w') as out_file:
    file_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['', 'plot_align' , 'similarity_align'])
    counter = 0
    for q in qa:
        movie_idx = q[4]
        question = q[1]
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        index = 0
        movie_align = ''
        sim_mtx = []
        comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
        sen_mtx = []
        for sen in story[movie_idx]:
            sen_mtx.append((sen,comparison_metric))
        sim_mtx = web_model.predict(sen_mtx)
        #sim_mtx_c = clinical_model.predict(sen_mtx)
        #sim_mtx = np.divide(sim_mtx_w,np.mean(sim_mtx_w)) + np.divide(sim_mtx_c,np.mean(sim_mtx_c))
        top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]
        for a in top_sentences:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n"))
        file_writer.writerow([counter, q[5], top_sentences])
        print("WEB",counter)
        correct_ans = q[3]
        counter = counter + 1

