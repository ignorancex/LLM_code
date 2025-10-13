from utils import *

import numpy as np


CHAINS = {
    'increase': '{p1} earns twice as much as {p2}.',
    'decrease': '{p1} earns half as much as {p2}.',
    'same': '{p1} earns the same salary as {p2}.',
    'independent': '{p1} received ${val} last week.'
}

QUESTIONS = {
    'single': 'How much salary did {p1} get?',
    'total': 'How much is the total salary of the referenced people?',
}

SYSTEM_PROMPT = """
You are a financial assistant AI skilled in calculating wages and solving salary-related queries.
I will give you context with the facts about salary of several people.
You need to answer the question based only on the information from the facts.
Before you derive the final answer, provide me a brief explanation.
Output your final verdict by strictly following this format: '## Answer: ${your_answer}' 
"""[1:-1]

TEMPLATE = """
There are {num_names} workers in the office.
Their names are as follows: {names}

Salary for each worker is as follows:
{context}

Now, respond to my question:
{question}
"""[1:-1]


def coin():
    return np.random.choice(['increase', 'same', 'decrease'])


def process_step1(names, k): # 대상 인물 지정
    return np.random.choice(names, k, replace=False)


def process_step2(names, val, chain_type): # chain type에 맞게 chain들 생성해주기
    chain_val_dict = {'increase': 2.0, 'same': 1, 'decrease': 0.5}
    if chain_type in ['parallel']:
        chain_ = []
        chain_val_ = []
        for i in range(len(names)):  # k = len(names)
            tmp_val = val * np.random.choice([0.125, 0.25, 0.5, 1, 2, 4, 8])
            tmp_chain = CHAINS['independent'].replace('{val}', str(tmp_val)).replace('{p1}', names[i])
            chain_val_.append(tmp_val)
            chain_.append(tmp_chain)
    elif chain_type in ['forward', 'backward', 'chaotic']:
        chain_ = [CHAINS['independent'].replace('{val}', str(val)).replace('{p1}', names[0])]
        chain_val_ = [val]
        for i in range(1, len(names)):
            c_ = coin()
            tmp_val = chain_val_[-1] * chain_val_dict[c_]
            tmp_chain = CHAINS[c_].replace('{p1}', names[i]).replace('{p2}', names[i - 1])
            chain_val_.append(tmp_val)
            chain_.append(tmp_chain)
    else:
        raise Exception('chaine type error')
    return chain_, chain_val_


def process_step3(names, chains, chain_vals, chain_type):  # chain들로부터 최종 데이터 만들어주기
    if chain_type in ['parallel', 'chaotic']:
        order = np.random.permutation(len(chains))
        last_name = names[-1]
        last_value = chain_vals[-1]
        chain_ = [chains[idx] for idx in order]
        chain_val_ = [chain_vals[idx] for idx in order]
        names_ = [names[idx] for idx in order]
    elif chain_type == 'forward':
        chain_ = chains
        chain_val_ = chain_vals
        names_ = names
        last_name = names[-1]
        last_value = chain_vals[-1]
    elif chain_type == 'backward':
        chain_ = chains[::-1]
        chain_val_ = chain_vals[::-1]
        names_ = names[::-1]
        last_name = names_[0]
        last_value = chain_val_[0]
    else:
        raise Exception('chain type error 22')
    return names_, chain_, chain_val_, last_name, last_value


def prepare_chain(idx, k=10, val=1600):
    names = process_step1(NAMES, k)

    parallel_chain_, parallel_chain_val_ = process_step2(names, val, 'parallel')
    forward_chain_, forward_chain_val_ = process_step2(names, val, 'forward')

    n_p, c_p, cv_p, ln_p, lv_p = process_step3(names, parallel_chain_, parallel_chain_val_, 'parallel')
    n_f, c_f, cv_f, ln_f, lv_f = process_step3(names, forward_chain_, forward_chain_val_, 'forward')
    n_b, c_b, cv_b, ln_b, lv_b = process_step3(names, forward_chain_, forward_chain_val_, 'backward')
    n_c, c_c, cv_c, ln_c, lv_c = process_step3(names, forward_chain_, forward_chain_val_, 'chaotic')

    if lv_f > (val * 64) or lv_f < (val / 64):
        return None
    return {
        'idx': idx,
        'names': ', '.join(names),
        'parallel_chain': '\n'.join(c_p),
        'parallel_total_val': sum(cv_p),
        'parallel_lastname': ln_p,
        'parallel_single_val': lv_p,
        'forward_chain': '\n'.join(c_f),
        'forward_total_val': sum(cv_f),
        'forward_lastname': ln_f,
        'forward_single_val': lv_f,
        'backward_chain': '\n'.join(c_b),
        'backward_total_val': sum(cv_b),
        'backward_lastname': ln_b,
        'backward_single_val': lv_b,
        'chaotic_chain': '\n'.join(c_c),
        'chaotic_total_val': sum(cv_c),
        'chaotic_lastname': ln_c,
        'chaotic_single_val': lv_c,
    }


def main(args):
    results_dir = args.results_dir
    k = args.k
    n = args.n
    val = args.val
    save_filename = os.path.join(results_dir, f'k{k}---val{val}.jsonl')
    data = []

    for i in range(n):
        chain = None
        while chain is None:
            # print(f'retry {i}')
            chain = prepare_chain(i, k, val)
        data.append(chain)

    write_jsonl(data, save_filename)
    print(f"data saved: {save_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='number of needles for each chain')
    parser.add_argument('--n', type=int, default=200, help='number of chain for each dataset')
    parser.add_argument('--val', type=int, default=1600, help='detail of needle')
    parser.add_argument('--results_dir', default='./data', help='data save path')
    args = parser.parse_args()

    main(args)
