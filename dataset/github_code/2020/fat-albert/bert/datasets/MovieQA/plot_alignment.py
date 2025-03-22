import config as cfg
import story_loader
import data_loader
import csv

mqa = data_loader.DataLoader()

def stringReplace(input):
    output = input
    if not output:
        output = 'None'
    return output

## LOAD THE TRAINING DATASET
movie_list = mqa.get_split_movies(split='train')
story, qa = mqa.get_story_qa_data('train', 'split_plot')

with open('data/aligned-train.csv', mode='w') as training_file:
    employee_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['', 'video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'ending4', 'label'])
    counter = 0
    for q in qa:
        movie_idx = q[4]
        movie_story = ', '.join(story[movie_idx])
        question = q[1]
        index = 0
        #story_split = movie_story.splitlines()
        movie_align = ''
        for a in q[5]:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n"))  + question
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        correct_ans = q[3]

        employee_writer.writerow([counter, movie_idx, counter, combined, movie_align, question, movie_align, ans_1, ans_2, ans_3, ans_4, ans_5,
              correct_ans])
        counter = counter + 1


## LOAD THE VALIDATION DATASET
movie_list = mqa.get_split_movies(split='val')
story, qa = mqa.get_story_qa_data('val', 'split_plot')
with open('data/aligned-val.csv', mode='w') as val_file:
    employee_writer = csv.writer(val_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['', 'video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'ending4', 'label'])
    counter = 0
    for q in qa:
        movie_idx = q[4]
        movie_story = ', '.join(story[movie_idx])
        question = q[1]
        index = 0
        #story_split = movie_story.splitlines()
        movie_align = ''
        for a in q[5]:
            movie_align = movie_align + " ".join(story[movie_idx][a].split("\n"))
        combined = " ".join(movie_align.split("\n")) + question
        ans_1 = stringReplace(q[2][0])
        ans_2 = stringReplace(q[2][1])
        ans_3 = stringReplace(q[2][2])
        ans_4 = stringReplace(q[2][3])
        ans_5 = stringReplace(q[2][4])
        correct_ans = q[3]

        employee_writer.writerow([counter, movie_idx, counter, combined, movie_align, question, movie_align, ans_1, ans_2, ans_3, ans_4, ans_5,
              correct_ans])
        counter = counter + 1