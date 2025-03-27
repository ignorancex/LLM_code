
import argparse
import json
from datetime import date
from tqdm import tqdm

# This code is modified from `evaluate.py` in https://github.com/shijx12/KQAPro_Baselines

'''
Usage:
    $ TEST_DATA_SET_PATH='path/to/test/data/set.json'
    $ PREDICTION_FILE_PATH='path/to/predictions.txt'
    $ python -m domain.kqapro.evaluate --auxiliary --except-yn --test $TEST_DATA_SET_PATH --pred $PREDICTION_FILE_PATH
'''

def whether_equal(answer, prediction):
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except Exception:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except Exception:
            return False

    answer = truncate_float(answer)
    prediction = truncate_float(prediction)
    if equal_as_date(answer, prediction):
        return True
    else:
        return answer == prediction


def load(f):
    data = []
    for line in f:
        data.append(json.loads(line.strip()))
    return data


def test(args):
    with open(args.test_file_path) as f:
        test_set = json.load(f)
    predictions = [x.strip() for x in open(args.pred_file_path).readlines()]  # one prediction per line
    with open(args.train_file_path) as f:
        train_set = json.load(f)
    train_answer_set = set(x['answer'] for x in train_set)

    labels = ['overall', 'multihop', 'qualifier', 'comparison', 'logical', 'count', 'verify', 'zero-shot']
    total = {k: 0 for k in labels}
    correct = {k: 0 for k in labels}

    for i in tqdm(range(len(predictions))):
        cur_labels = ['overall']
        functions = [f['function'] for f in test_set[i]['program']]

        for f in functions:
            if f in {'Relate'} or f.startswith('Filter'):
                cur_labels.append('multihop')
                break
        for f in functions:
            if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
                cur_labels.append('qualifier')
                break
        for f in functions:
            if f in {'SelectBetween', 'SelectAmong'}:
                cur_labels.append('comparison')
                break
        for f in functions:
            if f in {'And', 'Or'}:
                cur_labels.append('logical')
                break
        for f in functions:
            if f in {'Count'}:
                cur_labels.append('count')
                break
        for f in functions:
            if f in {'VerifyStr', 'VerifyNum', 'VerifyYear', 'VerifyDate'}:
                cur_labels.append('verify')
                break

        answer = test_set[i]['answer']
        if answer not in train_answer_set:
            cur_labels.append('zero-shot')

        if whether_equal(answer, predictions[i]):
            for k in cur_labels:
                correct[k] += 1
        else:
            pass
        for k in cur_labels:
            total[k] += 1

    for k in labels:
        print('{}: {:.2f}% ({}/{})'.format(k, correct[k]/total[k] * 100, correct[k], total[k]))
    if len(predictions) < len(test_set):
        print('WARNING: there are only {} predictions (need {})'.format(len(predictions), len(test_set)))


def auxiliary_test(args):
    with open(args.test_file_path) as f:
        test_set = json.load(f)
    predictions = [x.strip() for x in open(args.pred_file_path).readlines()]  # one prediction per line

    labels = ['overall']
    total = {k: 0 for k in labels}
    correct = {k: 0 for k in labels}

    for i in tqdm(range(len(predictions))):
        cur_labels = ['overall']

        choices = test_set[i]['choices']
        if args.except_yes_or_no_question and is_yes_or_no_question(choices):
            continue

        if any(whether_equal(choice, predictions[i]) for choice in choices):
            for k in cur_labels:
                correct[k] += 1

        for k in cur_labels:
            total[k] += 1

    for k in labels:
        print('{}: {:.2f}% ({}/{})'.format(k, correct[k]/total[k] * 100, correct[k], total[k]))

    if len(predictions) < len(test_set):
        print('WARNING: there are only {} predictions (need {})'.format(len(predictions), len(test_set)))


def is_yes_or_no_question(choices):
    return set(choices) == {'yes', 'no', 'unknown'}


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--auxiliary', dest='auxiliary', action='store_true', help='whether to perform auxiliary test')
    parser.add_argument('--except-yn', dest='except_yes_or_no_question', action='store_true',
                        help='whether to include yes-or-no questions for auxiliary test')
    parser.add_argument('--train', dest='train_file_path', help='a path to the train set file')
    parser.add_argument('--test', dest='test_file_path', help='a path to the test set file')
    parser.add_argument('--pred', dest='pred_file_path', help='a path to the prediction file')
    # auxiliary

    args = parser.parse_args()

    if args.auxiliary:
        auxiliary_test(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
