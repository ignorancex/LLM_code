from datasets import load_dataset
import numpy as np
from test_one_solution import check_correctness
from tqdm import tqdm

# check that the ids are from 0 to 4999
ids = [eg['problem_id'] for eg in load_dataset("codeparrot/apps")['test']]
assert(np.all(np.array(ids) == np.arange(5000)))
difficulties = [eg['difficulty'] for eg in load_dataset("codeparrot/apps")['test']]
for d in difficulties[:3000]:
    assert (d=='interview')
for d in difficulties[3000:4000]:
    assert (d=='competition')
for d in difficulties[4000:]:
    assert (d=='introductory')

# read generated codes
gen_codes_path = "../saved_models/apps_generation/codet5_lr_1e-05max_length_source_text_600max_length_target_code_512dfg_loss_weight_0.0ast_loss_weight_0.0num_beams_5train_batch_size_20/39999_output_pred.txt"
# gen_codes_path = "../saved_models/apps_generation/lr_1e-05max_length_source_text_600max_length_target_code_512dfg_loss_weight_0.1ast_loss_weight_0.1num_beams_5train_batch_size_20/189999_output_pred.txt"
generated_codes = open(gen_codes_path).readlines()
generated_codes = [code.strip().replace('--<NEWLINE>--', '\n') for code in generated_codes]

# test cases
data_dir = '../../datasets/APPS/test/'

# evaluate
def evaluate(start, end, desc):
    per_prob_res = []
    all_correct = []
    pbar = tqdm(range(start, end))
    for i in pbar:
        gen_code = generated_codes[i].replace('\\n', '\n').replace('\\"','"').replace('\\r','').replace('\\\\','\\').replace('\\t','\t')
        problem_results = check_correctness(prob_path=data_dir+str(i).zfill(4), generation=gen_code, 
                                            timeout=10, debug=False)
        problem_results = np.asarray(problem_results)
        per_prob_res.append(np.mean(problem_results > 0))
        all_correct.append(np.all(problem_results > 0))
        pbar.set_description(str(i)+', '+str(np.mean(per_prob_res)*100)+', '+str(np.mean(all_correct)*100))
    print ('='*10, desc, gen_codes_path.split('/')[-1].split('_')[0])
    per_prob_res, all_correct = round(np.mean(per_prob_res)*100,2), round(np.mean(all_correct)*100,2)
    print("Test Case Average (average accuracy over problems)", per_prob_res)
    print("Strict Accuracy (all test cases passed / total problems", all_correct)
    return (per_prob_res, all_correct)

# evaluate(4000,5000, 'introductory') # intro
# evaluate(3000,4000, 'competition') # competition
# evaluate(0,3000, 'interview') # interview

subsets = {'introductory':(4000,5000), 'competition':(3000,4000), 'interview':(0,3000)}
for k,(s,e) in subsets.items():
    subsets[k] = evaluate(s,e,k)
print (' '.join([str(subsets[k][0]) for k in ['introductory', 'interview', 'competition']]))
print (' '.join([str(subsets[k][1]) for k in ['introductory', 'interview', 'competition']]))






