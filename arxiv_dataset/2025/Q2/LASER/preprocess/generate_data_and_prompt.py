import json
import os
import pickle
from datetime import date
import random
import heapq
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10

lm_hist_max = 30

ml_group_map = {
    0: ['Adventure'],
    1: ['Comedy'],
    2: ['Action', 'Crime'],
    3: ['Drama'],
    4: ['Animation', 'Children'],
    5: ['Sci-Fi', 'Fantasy', 'Mystery'],
    6: ['Documentary'],
    7: ['Horror', 'Thriller'],
    8: ['Western', 'Film-Noir', 'Romance', 'War', 'Musical', 'IMAX', 'listed)'],
}
amz_group_map = {
    0: ['Literature & Fiction', 'Literature &amp; Fiction'],
    1: ['Mystery, Thriller & Suspense', 'Mystery, Thriller &amp; Suspense'],
    2: ['Science Fiction & Fantasy', 'Science Fiction &amp; Fantasy'],
    3: ['Teen & Young Adult', 'Teen &amp; Young Adult'],
    4: ['Christian Books & Bibles', 'Christian Books &amp; Bibles'],
    5: ['Romance'],
    6: ['Biographies & Memoirs', 'Biographies &amp; Memoirs', 'History', 'Reference'],
    7: ['New, Used & Rental Textbooks', 'New, Used &amp; Rental Textbooks',
          'Science &amp; Math', 'Science & Math',
          'Engineering &amp; Transportation', 'Engineering & Transportation',
          'Computers &amp; Technology', 'Computers & Technology',
          'Medical Books', 'Test Preparation'
          ],
    8: ['Politics & Social Sciences',  'Politics &amp; Social Sciences',
          'Lesbian, Gay, Bisexual & Transgender Books',
          'Lesbian, Gay, Bisexual &amp; Transgender Books',
          'Business & Money', 'Business &amp; Money',
          'Arts &amp; Photography', 'Arts & Photography',
          'Religion & Spirituality', 'Religion &amp; Spirituality',
          'Education & Teaching',  'Education &amp; Teaching', 'Law',
          ],
    9: ['Humor & Entertainment', 'Humor &amp; Entertainment',
          'Comics &amp; Graphic Novels', 'Comics & Graphic Novels',
          'Parenting & Relationships', 'Parenting &amp; Relationships',
          'Crafts, Hobbies & Home', 'Crafts, Hobbies &amp; Home',
          'Health, Fitness & Dieting', 'Health, Fitness &amp; Dieting',
          'Sports &amp; Outdoors', 'Sports & Outdoors',
          'Cookbooks, Food & Wine', 'Cookbooks, Food &amp; Wine',
          'Children\'s Books',  'unknown category',  'Calendars', 'Self-Help', 'Travel',
          ]
}
# {'0': 23161, '7': 997, '2': 5182, '9': 6453, '4': 4903, '1': 11595, '8': 2210, '6': 3013, '5': 16772, '3': 4032}


def save_prompts(file_name, prompt_list):
    with open(file_name, "w") as json_file:
        for dictionary in prompt_list:
            json.dump(dictionary, json_file)
            json_file.write('\n')


def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)): # start idx 后面的
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))
    print(full_data[:5])
    return full_data


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name,
                         framework, recent=False):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name in ['ml-1m', 'ml-10m', 'ml-25m', 'ml-10m-new']:
        attr_idx = 0
        # user2attribute = datamap['user2attribute']
    else:
        attr_idx = 1

    hist_prompts_list = []
    print('item2attribute', list(item2attribute.items())[:10])
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        if recent:
            hist_item_seq = item_seq[:cur_idx]
            hist_rating_seq = rating_seq[:cur_idx]
        else:
            hist_item_seq = item_seq[:cur_idx-1]
            hist_rating_seq = rating_seq[:cur_idx-1]
        hist_prompts = {}
        if framework == 'KAR':
            historys = []
            for iid, rating in zip(hist_item_seq, hist_rating_seq):
                tmp = '"{}", {} stars; '.format(itemid2title[str(iid)], int(rating))
                historys.append(tmp)
            if dataset_name in ['amz', 'amz-new']:
                question = 'Analyze user\'s preferences on books about factors like genre, author, writing style, ' \
                           'theme, setting, length and complexity, time period, literary quality, critical ' \
                           'acclaim (Provide clear explanations based on relevant details from the user\'s book '\
                           'viewing history and other pertinent factors.'

                prompt = 'Given user\'s book rating history: ' + ''.join(historys) + question
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
                # gender, age, occupation = user2attribute[uid]
                # user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                #             ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                #                                         OCCUPATION_MAPPING[occupation])
                # 与kar的prompt略有改动
                question = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                        'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                        'and soundtrack). Please provide clear and brief explanations based on relevant details from the user\'s movie ' \
                        'viewing history and other pertinent factors.'
                # prompt2: LLM生成的内容更简洁
                # question = ("Based on the information about movies that the user has interacted, you are required to "
                #             "generate user profile. Please provide a summarization of what types of movie this user is "
                #             "likely to enjoy and briefly explain your reasoning for the summarization.")

                # 记录user id
                prompt = 'Given user\'s movie rating history: ' + ''.join(historys) + question
            else:
                raise NotImplementedError
        elif framework == 'TRAWL':
            descriptions = []
            historys = []
            for iid, rating in zip(hist_item_seq, hist_rating_seq):
                tmp = '{}, {} stars; '.format(itemid2title[str(iid)], int(rating))
                historys.append(tmp)
                cate = item2attribute[str(iid)][attr_idx]
                descriptions.append(attrid2name[str(cate)] + '; ')
            history_text = ''.join(historys)
            description_text = ''.join(descriptions)
            if dataset_name in ['amz', 'amz-new']:
                prompt = (f'Given user\'s book rating history:\n{history_text}\nDescriptions of the books in the '
                          f'user\'s viewing history are listed below.\n{description_text}\nAnalyze user\'s '
                          f'preferences on books (consider factors like genre, author, writing style, theme, '
                          f'critical acclaim). Provide clear explanations based on relevant details from the user\'s'
                          f'book viewing history and other pertinent factors.')
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
                # gender, age, occupation = user2attribute[uid]
                # user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                #             ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                #                                         OCCUPATION_MAPPING[occupation])
                prompt = (f'Given user\'s movie rating history:\n{history_text}\nDescriptions of '
                          f'the movies in the user\'s viewing history are listed below.'
                          f'\n{description_text}\nAnalyze user\'s preferences on movies (consider '
                          f'factors like genre, director/cast, plot/theme, critical acclaim/award). '
                          f'Provide clear explanations based on relevant details from the user\'s movie '
                          f'viewing history and other pertinent factors.')
            else:
                raise NotImplementedError
        elif framework == 'RLMRec':
            if dataset_name in ['amz', 'amz-new']:
                system_prompt = """You will serve as an assistant to help me determine which types of books a specific user is likely to enjoy.
I will provide you with information about books that the user has purchased, as well as his or her reviews of those books.
Here are the instructions:
1. Each purchased book will be described in JSON format, with the following attributes:
{
    "title": "the title of the book", (if there is no title, I will set this value to "None")
    "description": "a description of what types of users will like this book",
    "rating": "the user's rating on the book"
}

2. The information I will give you:
PURCHASED ITEMS: a list of JSON strings describing the books that the user has rating.

Requirements:
1. Please provide your decision in JSON format, following this structure:
{
    "summarization": "A summarization of what types of books this user is likely to enjoy" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}
2. Please ensure that the "summarization" is no longer than 100 words.
3. The "reasoning" has no word limits.
4. Do not provided any other text outside the JSON string.
"""
                historys = []
                for iid, rating in zip(hist_item_seq, hist_rating_seq):
                    title = itemid2title[str(iid)]
                    rating = int(rating)
                    cate = item2attribute[str(iid)][attr_idx]
                    json_dict = {
                        'title': title,
                        'rating': f'{rating} stars',
                    }
                    historys.append(json.dumps(json_dict))
                history_text = '\n'.join(historys)
                prompt = system_prompt + history_text + '\n'
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
                system_prompt = """You will serve as an assistant to help me determine which types of movies a specific user is likely to enjoy.
I will provide you with information about movies that the user has purchased, as well as his or her reviews of those movies.
Here are the instructions:
1. Each purchased movie will be described in JSON format, with the following attributes:
{
    "title": "the title of the movie", (if there is no title, I will set this value to "None")
    "description": "a description of what types of users will like this movie",
    "rating": "the user's rating on the movie"
}

2. The information I will give you:
PURCHASED ITEMS: a list of JSON strings describing the movies that the user has rated.

Requirements:
1. Please provide your decision in JSON format, following this structure:
{
    "summarization": "A summarization of what types of movies this user is likely to enjoy" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}
2. Please ensure that the "summarization" is no longer than 100 words.
3. The "reasoning" has no word limits.
4. Do not provided any other text outside the JSON string.
"""
                historys = []
                for iid, rating in zip(hist_item_seq, hist_rating_seq):
                    title = itemid2title[str(iid)]
                    rating = int(rating)
                    cate = item2attribute[str(iid)][attr_idx]
                    json_dict = {
                        'title': title,
                        'rating': f'{rating} stars',
                    }
                    historys.append(json.dumps(json_dict))
                history_text = '\n'.join(historys)
                prompt = system_prompt + history_text + '\n'
            else:
                raise NotImplementedError
        elif framework == 'ONCE':
            historys = []
            for iid, rating in zip(hist_item_seq, hist_rating_seq):
                tmp = '{}, {} stars; '.format(itemid2title[str(iid)], int(rating))
                historys.append(tmp)
            history_text = ''.join(historys)
            if dataset_name in ['amz', 'amz-new']:
                prompt = (f"You are asked to describe user interest based on his/her browsed book list listed below:"
                          f"\n\n {history_text}\n\nYou can only response the user interests in a single word or phrase"
                          f" (at most 5). You are not allowed to response any other words for any explanation or note.")
                # output_format = """
                # {
                #     "title": ...,
                #     "description": ...,
                # }
                #                 """
                # prompt = (f"You are asked to capture user's interest based on his/her browsing history, and "
                #           f"recommend a book that he/she may be interested. The history is as below:"
                #           f"\n\n {history_text}\n\nYou can only recommend one book (only one) in the following "
                #           f"json format: {output_format}The book should be diverse, that is not too similar with the "
                #           f"original provided book list. You are not allowed to response any other words for any "
                #           f"explanation or note. JUST GIVE ME JSON-FORMAT BOOK.")
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
                prompt = (f"You are asked to describe user interest based on his/her browsed movie list listed below:"
                          f"\n\n {history_text}\n\nYou can only response the user interests in a single word or phrase"
                          f" (at most 5). You are not allowed to response any other words for any explanation or note.")
                # output_format = """
                # {
                #     "title": ...,
                #     "description": ...,
                # }
                #                 """
                # prompt = (f"You are asked to capture user's interest based on his/her browsing history, and "
                #           f"recommend a movie that he/she may be interested. The history is as below:"
                #           f"\n\n {history_text}\n\nYou can only recommend one movie (only one) in the following "
                #           f"json format: {output_format}The movie should be diverse, that is not too similar with the "
                #           f"original provided movie list. You are not allowed to response any other words for any "
                #           f"explanation or note. JUST GIVE ME JSON-FORMAT MOVIE.")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        hist_prompts["question_id"] = user
        hist_prompts['prompt'] = prompt
        hist_prompts_list.append(hist_prompts)

    print('data num', len(hist_prompts_list))
    print(hist_prompts_list[0])
    return hist_prompts_list


def generate_item_prompt(iids, item2attribute, datamap, dataset_name, framework):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = []
    # for iid, title in itemid2title.items():
    print('current framework', framework)
    for iid in iids:
        title = itemid2title[iid]
        item = id2item[iid]
        item_prompt = {}
        if framework == 'KAR':
            if dataset_name in ['amz', 'amz-new']:
                brand, cate = item2attribute[str(iid)]
                brand_name = attrid2name[str(brand)]
                # cate_name = attrid2name[cate]
                prompt = 'Introduce book {}, which is from brand {} and describe its attributes including but' \
                                    ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                    'time period, literary quality, critical acclaim.'.format(title, brand_name)
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new', 'ml-25m']:
                prompt = 'Introduce movie {} and describe its attributes (including but not limited to genre, ' \
                                    'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                    'acclaim/award, production quality, and soundtrack).'.format(title)
            else:
                raise NotImplementedError
        elif framework == 'RLMRec':
            if dataset_name in ['amz', 'amz-new']:
                brand, cate = item2attribute[str(iid)]
                brand_name = attrid2name[str(brand)]
                cate_name = attrid2name[str(cate)]
                system_prompt = """You will serve as an assistant to help me summarize which types of users would enjoy a specific book.
I will provide you with the title and a description of the book.
Here are the instructions:
1. I will provide you with information in the form of a JSON string that describes the book:
{
    "title": "the title of the book", (if there is no title, I will set this value to "None")
    "description": "a description of the book", (if there is no description, I will set this value to "None")
}

Requirements:
1. Please provide your answer in JSON format, following this structure:
{
    "summarization": "A summarization of what types of users would enjoy this book" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}
2. Please ensure that the "summarization" is no longer than 200 words.
3. Please ensure that the "reasoning" is no longer than 200 words.
4. Do not provide any other text outside the JSON string.
"""
                input = {
                    'title': title,
                    'description': f'category is {cate_name}, brand is {brand_name}'
                }
                prompt = system_prompt + json.dumps(input) + '\n'
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
                cate = item2attribute[str(iid)][0]
                cate_name = attrid2name[str(cate)]
                system_prompt = """You will serve as an assistant to help me summarize which types of users would enjoy a specific movie.
I will provide you with the title and a description of the movie.
Here are the instructions:
1. I will provide you with information in the form of a JSON string that describes the movie:
{
    "title": "the title of the movie", (if there is no title, I will set this value to "None")
    "description": "a description of the movie", (if there is no description, I will set this value to "None")
}

Requirements:
1. Please provide your answer in JSON format, following this structure:
{
    "summarization": "A summarization of what types of users would enjoy this movie" (if you are unable to summarize it, please set this value to "None")
    "reasoning": "briefly explain your reasoning for the summarization"
}
2. Please ensure that the "summarization" is no longer than 200 words.
3. Please ensure that the "reasoning" is no longer than 200 words.
4. Do not provide any other text outside the JSON string.
                """
                input = {
                    'title': title,
                    'description': f'category is {cate_name}'
                }
                prompt = system_prompt + json.dumps(input) + '\n'
            else:
                raise NotImplementedError
        elif framework == 'TRAWL':
            if dataset_name in ['amz', 'amz-new']:
                brand, cate = item2attribute[str(iid)]
                brand_name = attrid2name[str(brand)]
                cate_name = attrid2name[str(cate)]
                prompt = (f'Your task is to generate a description for a book, given its title and relevant text.\n'
                          f'The title is: {title}.\nNext is the relevant text about the work: category is {cate_name}, '
                          f'brand is {brand_name}.\nBased on the information above, describe the attributes of the '
                          f'work (including, but not limited to genre, author, writing style, theme, critical acclaim).')
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new', 'ml-25m']:
                cate = item2attribute[str(iid)][0]
                cate_name = attrid2name[str(cate)]
                prompt = (f'Your task is to generate a description for a movie, given its title and relevant text.\n'
                          f'The title is: {title}.\nNext is the relevant text about the work: category is {cate_name}.'
                          f'\nBased on the information above, describe the attributes of the work (including, '
                          f'but not limited to genre, director/cast, plot/theme, critical acclaim/award).')
            else:
                raise NotImplementedError
        # elif framework == 'RLMRec':
        #     if dataset_name in ['amz', 'amz-new']:
        #         pass
        #     elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new']:
        #         pass
        #     else:
        #         raise NotImplementedError
        elif framework == 'ONCE':
            if dataset_name in ['amz', 'amz-new']:
                system = ("You are asked to act as a book description generator. I will provide you the title of "
                "a book, which format is as below:\n\n[book] {title}\n\nwhere {title} will be filled with content. " 
                "You can only response the description of this book in English which should be clear, concise, " 
                "objective and neutral. You are not allowed to response any other words for any explanation. "
                "Your response format should be:\n\n[description] {description}\n\nwhere {description} should be "
                "filled with the book description. Now, your role of book description generator formally begins. "
                "Any other information should not disturb your role.\n\n")
                prompt = system + f'[book] {title}\n\n'
            elif dataset_name in ['ml-1m', 'ml-10m', 'ml-10m-new', 'ml-25m']:
                system = ("You are asked to act as a movie description generator. I will provide you the name and "
                "released year of a movie, which format is as below:\n\n[movie] {name} (year)\n\nwhere {name} will be " 
                "filled with the movie name. You can only response the detailed description about this movie in "
                "English which should be clear, concise, objective and neutral. You are not allowed to response any " 
                "other words for any explanation. Your response format should be:\n\n[description] {description}\n\n"
                "where {description} should be filled with the movie description. Now, your role of movie description" 
                "generator formally begins. Any other information should not disturb your role.\n\n")
                prompt = system + f'[movie] {title}\n\n'
        else:
            raise NotImplementedError
        item_prompt['prompt'] = prompt
        item_prompt['question_id'] = item
        item_prompts.append(item_prompt)
    print('data num', len(item_prompts))
    print(item_prompts[0])
    return item_prompts


def get_random_group(iids, n_clusters, id2item):
    item_num = len(iids)
    random.shuffle(iids)
    avg = item_num // n_clusters
    remainder = item_num % n_clusters

    sublists = [iids[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)]
                for i in range(n_clusters)]

    item2class = {id2item[str(i)]: j for j, sublist in enumerate(sublists) for i in sublist}
    return item2class


def split_and_save_item_prompts(item2attribute, datamap, dataset_name, data_dir, framework):
    id2attribute = datamap['id2attribute']
    split_file = os.path.join(data_dir, 'item_split.json')
    if os.path.exists(split_file):
        data = load_json(split_file)
        iids = data['iids']
        recent_iids = data['recent_iids']
    else:
        attribute2iids = defaultdict(list)
        group2iids = defaultdict(list)
        attribute2num = defaultdict(int)
        group2num = defaultdict(int)
        id2item = datamap['id2item']
        if dataset_name in ['ml-1m', 'ml-10m', 'ml-25m', 'ml-10m-new']:
            attr_idx = 0
            group_map = ml_group_map
        else:
            attr_idx = 1
            group_map = amz_group_map
        attr2group = {v: k for k, v_list in group_map.items() for v in v_list}
        # print('Medical Books' in attr2group, print(group_map['7']))
        # print(attr2group)
        # print('attr2group', attr2group)
        for iid, attribute in item2attribute.items():
            attribute2iids[attribute[attr_idx]].append(iid)
            group2iids[attr2group[id2attribute[str(attribute[attr_idx])]]].append(iid)
            attribute2num[id2attribute[str(attribute[attr_idx])]] += 1
            group2num[attr2group[id2attribute[str(attribute[attr_idx])]]] += 1
        # for k, v in attribute2num.items():
        #     print(k, v)
        # exit()
        print(group2num)
        item2class = {id2item[iid]: attr2group[id2attribute[str(attr[attr_idx])]] for iid, attr in item2attribute.items()}
        save_json(item2class, data_dir + '/item_class.json')
        class2attri = {i: attr for attr, i in attr2group.items()}
        save_json(class2attri, data_dir + '/class2attribute.json')

        iids, recent_iids = [], []
        for group in group2iids:
            iid_list = group2iids[group]
            print(group, group_map[group], len(iid_list))
            random.shuffle(iid_list)
            mid = len(iid_list) // 2
            iids.extend(iid_list[: mid])
            recent_iids.extend(iid_list[mid:])
        random.shuffle(iids)
        random.shuffle(recent_iids)
        print('iids', len(iids), 'recent_iids', len(recent_iids))
        data = {
            'iids': iids,
            'recent_iids': recent_iids
        }
        save_json(data, os.path.join(data_dir, 'item_split.json'))

        item2random_group = get_random_group(iids + recent_iids, len(group_map), id2item)
        save_json(item2random_group, data_dir + '/item_class_random.json')

    item_prompts = generate_item_prompt(iids, item2attribute, datamap, dataset_name, framework)
    save_prompts(data_dir + f'/{framework}/item.json', item_prompts)

    recent_item_prompts = generate_item_prompt(recent_iids, item2attribute, datamap, dataset_name, framework)
    save_prompts(data_dir + f'/{framework}/recent_item.json', recent_item_prompts)


def behavioral_info_positive_sampling(sequence_data, uid_set, data_path):
    iid_uids = defaultdict(list)
    uid_iids = {}
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        uid_iids[uid] = item_seq
        for iid in item_seq:
            iid_uids[iid].append(uid)
    uid_iids = {k: set(v) for k, v in uid_iids.items()}
    iid_uids = {k: set(v) for k, v in iid_uids.items()}
    iid_set = set(iid_uids.keys())
    uu_scores = defaultdict(dict)
    # SWING score for every user-user pair, here we simply SWING to reduce complexity
    for uid1 in uid_set:
        for uid2 in uid_set:
            if uid1 != uid2:
                uid1_seq = uid_iids[uid1]
                uid2_seq = uid_iids[uid2]
                joint_iid = uid1_seq & uid2_seq
                if len(joint_iid) > 1:
                    uu_scores[uid1][uid2] = len(joint_iid)
                # score = 0
                # for iid1 in joint_iid:
                #     for iid2 in joint_iid:
                #         if iid1 != iid2:
                #             joint_uid = iid_uids[iid1] & iid_uids[iid2]
                #             score += 1 / (len(joint_uid))
                # uu_scores[uid1][uid2] = score
    print('get user SWING score')
    # SWING score for every item-item pair
    ii_scores = defaultdict(dict)
    for iid1 in iid_set:
        for iid2 in iid_set:
            if iid1 != iid2:
                iid1_seq = iid_uids[iid1]
                iid2_seq = iid_uids[iid2]
                joint_uid = iid1_seq & iid2_seq
                if len(joint_uid) > 1:
                    ii_scores[iid1][iid2] = len(joint_uid)
                # score = 0
                # for uid1 in joint_uid:
                #     for uid2 in uid2_seq:
                #         if uid1 != uid2:
                #             joint_iid = uid_iids[uid1] & uid_iids[uid2]
                #             score += 1 / (len(joint_iid))
                # ii_scores[iid1][iid2] = score
    print('get item SWING score')
    # find users with top SWING scores for each user
    uid_pos_pair = {}
    for uid in uid_set:
        top_users = heapq.nlargest(20, uu_scores[uid], key=uu_scores[uid].get)
        uid_pos_pair[uid] = top_users
    # find items with top SWING scores for each item
    iid_pos_pair = {}
    for iid in iid_set:
        top_items = heapq.nlargest(20, ii_scores[iid], key=ii_scores[iid].get)
        iid_pos_pair[iid] = top_items

    score_dict = {
        'pos_item_pair': iid_pos_pair,
        'pos_user_pair': uid_pos_pair,
    }
    save_json(score_dict, os.path.join(data_path, 'swing_scores.json'))
    print('Save score to', os.path.join(data_path, 'swing_scores.json'))


if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    DATA_SET_NAME = 'amz-new'
    # DATA_SET_NAME = 'ml-10m-new'
    # FRAMEWORK = 'KAR'  # KAR, LLM-Rec, RLMRec, TRAWL
    # FRAMEWORK = 'TRAWL'  # KAR, LLM-Rec, RLMRec, TRAWL
    # FRAMEWORK = 'ONCE'  # KAR, LLM-Rec, RLMRec, TRAWL
    FRAMEWORK = 'RLMRec'  # KAR, LLM-Rec, RLMRec, TRAWL
    if DATA_SET_NAME in ['ml-10m-new']:
        rating_threshold = 3
    else:
        rating_threshold = 4
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'])
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'])
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    print('generating reranking train dataset')
    train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                        train_test_split['train'], item_set)
    print('generating reranking test dataset')
    test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                       train_test_split['test'], item_set)
    print('save reranking data')
    save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    train_rerank, test_rerank = None, None

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')


    if FRAMEWORK == 'TRAWL':
        behavioral_info_positive_sampling(sequence_data, list(sequence_data.keys()), PROCESSED_DIR)
    os.makedirs(os.path.join(PROCESSED_DIR, FRAMEWORK), exist_ok=True)
    # print('generating item prompt')
    # 划分用户分组，保存用户prompt
    split_and_save_item_prompts(item2attribute, datamap, DATA_SET_NAME, PROCESSED_DIR, FRAMEWORK)

    print('generating history prompt')
    # 用户曾经的历史
    hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
                                       train_test_split['lm_hist_idx'],
                                       DATA_SET_NAME, FRAMEWORK)
    save_prompts(PROCESSED_DIR + f'/{FRAMEWORK}/history.json', hist_prompt)

    # 用户最新的历史
    recent_hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
                                              train_test_split['lm_hist_idx'],
                                              DATA_SET_NAME, FRAMEWORK, recent=True)

    save_prompts(PROCESSED_DIR + f'/{FRAMEWORK}/recent_history.json', recent_hist_prompt)


