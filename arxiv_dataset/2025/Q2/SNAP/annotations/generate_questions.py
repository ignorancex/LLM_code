# This script generates multiple-choice questions for the sensor bias dataset
# for each image we ask the following questions
# Open-ended
# Q1 Objects of what class are in the image? 
# Q2 How many objects are in the image?

# MC
# Q3 What class of object is in the image? 11 answer options
# Q4 How many objects are in the image? A) 2; B) 3; C) 4; D) 5

# python3 annotations/generate_questions.py

import os
import json
import random
import pandas as pd
import numpy as np

seed = 42

random.seed(seed)

data_ver='data_v5'

#sensor_bias_data_df = pd.read_excel(f'annotations/sensor_bias_{data_ver}.xlsx')
sensor_bias_data_df = pd.read_excel(f'annotations/SNAP.xlsx')

# to generate a smaller sample for testing
# data_df = data_df.sample(n=5000, random_state=seed)

object_categories = [x.replace('_', ' ') for x in sensor_bias_data_df['Category'].unique().tolist()]

int2str = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
alt_ans = {'comic book': 'comic, book, comic book',
			'mouse': 'mice, mouse',
			'water bottle': 'water bottle, bottle',
			'cup': 'cup, mug',
			'backpack': 'backpack',
			'phone': 'cellular, cell, smartphone, phone',
			'remote': 'remote, controller',
			'tie': 'tie, necktie', 
			'laptop': 'laptop, computer, notebook'
			}

def generate_questions(data_df, json_path):

	questions = {}
	for img_path, obj_cat, num_obj in zip(data_df['Path'], data_df['Category'], data_df['Num_objects']):

		questions[img_path] = {}

		questions[img_path]['Q1'] = 'Objects of what class are in the image? Answer with the name of the class.'
		if obj_cat_str in alt_ans:
			questions[img_path]['A1'] = alt_ans[obj_cat_str]
		else:
			questions[img_path]['A1'] = obj_cat_str

		questions[img_path]['Q2'] = 'How many objects are in the image? Answer with one number.'
		questions[img_path]['A2'] = f'{num_obj}, {int2str[num_obj]}'

		answer_options = ', '.join(random.sample(object_categories, len(object_categories)) + ['other'])
		questions[img_path]['Q3'] = f'Objects of what class are in the image? Select one of the following options: {answer_options}'
		questions[img_path]['A3'] = obj_cat_str

		shuffled_answers = random.sample([2,3,4,5], 4)
		letter_answers = ['A', 'B', 'C', 'D']
		answer_options = [[x[0], x[1]] for x in zip(letter_answers, shuffled_answers)]

		questions[img_path]['Q4'] = f'How many objects are in the image? Select one of the following options: {"    ".join([f"{x[0]}) {x[1]}" for x in answer_options])}'
		ans_idx = shuffled_answers.index(num_obj)
		questions[img_path]['A4'] = f'{answer_options[ans_idx]}'

		shuffled_answers = random.sample([2,3,4,5], 4)
		ans_idx = shuffled_answers.index(num_obj)
		letter_answers = ['A', 'B', 'C', 'D']
		answer_options = [[x[0], x[1]] for x in zip(letter_answers, shuffled_answers)]

	with open(json_path, 'w') as fid:
		json.dump(questions, fid, indent=4)


generate_questions(sensor_bias_data_df, f'annotations/questions_SNAP.json')