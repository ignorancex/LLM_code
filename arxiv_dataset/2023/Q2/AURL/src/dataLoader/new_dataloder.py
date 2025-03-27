import json
import random
import re
from torch.utils.data.dataloader import DataLoader
from src.dataLoader.dataset import ManagerDataset2
from src.dataLoader.utils_dataloader import InputFeature

logger_dic = {}
def read_file_from_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    for ind, dialog in data_json.items():
        data_iter = {'goal_value': json.loads(dialog['goal_value']), 'first_turn': None}
        user_iter = {
            'utterance': [],
            'user_label': [],
            'user_state': [],
            'tag': []
        }
        sys_iter = {
            'utterance': [],
            'staff_label': [],
            'staff_state': []
        }
        fst_turn = dialog['turns'][0]
        data_iter['first_turn'] = 'user' if 'user' in fst_turn.keys() else 'sys'
        if data_iter['first_turn'] == 'sys':
            dialog['turns'].insert(0, {
                'user': "",
                'user_label': "<PAD>",
                'user_state': json.dumps([]),
                'tag': None
            })
        if 'staff' not in dialog['turns'][-1]:
            dialog['turns'].append({
                'staff': "再见",
                'staff_label': 'bye',
                'staff_state': dialog['turns'][-1]["user_state"]
            })
        for turn in dialog['turns']:
            if 'user' in turn.keys():
                utterance = turn['user']
                # user_iter['utterance'].append(utterance)
                user_iter['user_label'].append(turn['user_label'].lower())
                user_iter['user_state'].append(json.loads(turn['user_state']))
                user_iter['tag'].append(turn['tag'])
                if 'paraphrase' in turn:
                    paraphrase = turn['paraphrase']
                    user_label = turn['user_label']
                    if '(' not in user_label:
                        user_label += '()'
                    user_action = re.findall(r'(.*)[(]', user_label)[0]
                    user_slot = re.findall(r'[(](.*?)[)]', user_label)[0]
                    if user_action == 'update':
                        if paraphrase.find(user_slot) != -1:
                            utterance = paraphrase
                user_iter['utterance'].append(utterance)

            else:
                utterance = turn['staff']
                if 'paraphrase' in turn:
                    utterance = turn['paraphrase']
                sys_iter['utterance'].append(utterance)
                sys_iter['staff_label'].append(turn['staff_label'].lower())
                sys_iter['staff_state'].append(json.loads(turn['staff_state']))

        data_iter['user'] = user_iter
        data_iter['sys'] = sys_iter

        data.append(data_iter)
    return data


def read_dialogue_from_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    return data_json


def split_data(data, shuffle, partition):
    if shuffle:
        random.shuffle(data)
    total_partition = 0
    for i in partition:
        total_partition += i

    sub_lists = []
    ind = 0
    for pcg in partition:
        sub_lists.append(data[ind: int(len(data) * (ind + (pcg / total_partition)))])
        ind += int(len(data) * (pcg / total_partition))

    return sub_lists

def read_langs(file, voc):
    datas = []
    max_resp_len, max_value_len = 0, 0
    for name, data in enumerate(file):
        dialog_history = ""
        goal_value = data['goal_value']
        dialogue = {
            "user_utterance": [],
            "user_action": [],
            "user_slot": [],
            "tag": [],
            "staff_utterance": [],
            "staff_action": [],
            "staff_slot": [],
            "staff_state": [],
            "staff_slot_idx": [],
            "last_staff_state": [],
            "last_staff_action": [],
            "last_staff_slot": [],
            "dialog_history": [],
            "goal_value": [],
            "dst_target": [],
            "last_state_chunk_id": [],
            "word_target": [],
            "confirm_id": [],
            "reg_exp": []
        }
        for ti in range(len(data['sys']['utterance'])):
            user_utterance = data['user']['utterance'][ti]
            user_label = data['user']['user_label'][ti]
            user_state = data['user']['user_state'][ti]
            tag = data['user']['tag'][ti]

            staff_utterance = data['sys']['utterance'][ti]
            staff_label = data['sys']['staff_label'][ti]
            staff_state = data['sys']['staff_state'][ti]

            last_staff_state = data['sys']['staff_state'][ti - 1] if ti > 0 else []
            last_staff_label = data['sys']['staff_label'][ti - 1] if ti > 0 else '<PAD>'

            feature = InputFeature(user_utterance,
                                   user_label,
                                   user_state,
                                   tag,
                                   staff_utterance,
                                   staff_label,
                                   staff_state,
                                   last_staff_state,
                                   last_staff_label,
                                   dialog_history,
                                   goal_value)
            # dialog_history += user_utterance + ';'
            dialog_history = staff_utterance + ';'
            for keys, value in feature.getdict().items():
                dialogue[keys].append(value)

        dialogue['dial_len'] = len(dialogue['staff_utterance'])
        datas.append(dialogue)

    staff_dataset = ManagerDataset2(voc, datas)

    return staff_dataset

def json_print(feature, name):
    dic = {
        'user_utterance': feature.user_utterance,
        'staff_state': str(feature.last_staff_state),
        'staff_target': str(feature.staff_state),
        'word_attn': str(feature.word_target),
        'dst_target': str(feature.dst_target),
        'last_staff_slot': str(feature.last_staff_slot),
        'confirm_id': str(feature.confirm_id)
    }
    logger_dic[name].append(dic)


def data_loader(file_path, vocab_path, batch_size):
    raw_data = read_file_from_json(file_path)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    staff_dataset = read_langs(raw_data, vocab)
    staff_dataloader = DataLoader(staff_dataset, batch_size=batch_size, collate_fn=staff_dataset.collate_fn, shuffle=True)

    return staff_dataloader

if __name__ == '__main__':
    staff_dataloader = data_loader('../../data/new_data_手机号/para_train.json', '../corpus/vocabulary.json', 32)
    for batch in staff_dataloader:
        print(batch)
