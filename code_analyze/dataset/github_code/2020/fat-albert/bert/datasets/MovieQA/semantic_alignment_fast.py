import config as cfg
import story_loader
import data_loader
import csv
import numpy as np
from semantic_text_similarity.models import WebBertSimilarity

web_model = WebBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
num_sentences = 6
mqa = data_loader.DataLoader()

def stringReplace(input):
    output = input
    if not output:
        output = 'None'
    return output

## LOAD THE TEST DATASET
movie_list = mqa.get_split_movies(split='test')
story, qa = mqa.get_story_qa_data('test', 'split_plot')
with open('data/test.csv', mode='w') as test_file:
    employee_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['', 'video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'ending4', 'label'])

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
        sim_mtx = []
        comparison_metric = question + ans_1 + ans_2 + ans_3 + ans_4 + ans_5
        sen_mtx = []
        for sen in story[movie_idx]:
            sen_mtx.append((sen,comparison_metric))
            #sim_mtx.append(web_model.predict([(sen, comparison_metric)]).item())
        sim_mtx = web_model.predict(sen_mtx)
        top_sentences = np.asarray(sim_mtx).argsort()[-num_sentences:][::-1]
        print(top_sentences)
        for a in top_sentences:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n"))  + question
        correct_ans = 0

        employee_writer.writerow([counter, movie_idx, counter, combined, movie_align, question, movie_align, ans_1, ans_2, ans_3, ans_4, ans_5,
              correct_ans])
        counter = counter + 1
