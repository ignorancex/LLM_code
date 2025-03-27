import subprocess
import os
import threading
import traceback
import json
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from os.path import join as pjoin

# GLOBAL
# spack load /u64kqpe /tymduoi && pip install -U dataset
rep_correct = 'All Repaired'
rep_incorrect = 'Old Repairs were incomplete, apply these repairs too:'
rep_error = 'There are issues with the suggested repairs. The new program may not run.'
rep_not_needed = 'No repair!'
rep_partial = 'Partial Repaired'
timeout = 'TIMEOUT'

graph_matching_options = [0, 1, 3]
loc_add = 'Locs Added:'
loc_del = 'Locs Deleted:'
loc_same = 'Locs Same'

test_available = 'Test Case Available'
test_not_available = 'Test Case Not Available'
path = '/data/ScrapeData/'
corr_locs = 'Locs in Correct Program Model'
exit_code_text = 'Exitcode:'
corr_exp = 'Exprs in Correct Program Model'
incorr_locs = 'Locs in Incorrect Program Model'
incorr_exps = 'Exprs in Incorrect Program Model'
old_incorr_locs = 'Locs in Old Incorrect Program Model'
old_incorr_exps = 'Exprs in Old Incorrect Program Model'
num_Reps = 'Number of Repairs '
# docker run --name=clara_container -v E:\code\Clara_Data\Output:/data -it clara

def remove_unicode(fname):
    """ Run on just one file.
    """
    source = open(fname, 'r', encoding='utf-8')
    dir_name = os.path.dirname(fname)
    pname = os.path.basename(fname).split("_")[0]
    with open(os.path.join(dir_name,f'{pname}_solution.py'), 'w', encoding='utf-8') as mod:
        r = source.read().replace('\u00a0','')
        mod.write(r)
    source.close()


class ClaraResults():
    def __init__(self):
        self.data = {}
        self.row = {
            "Correct File": "", 
            "Incorrect File": "",
            "Repair": "",
            "Structure Mismatch": "",
            "Repair Correct": "",
            "Match": "",
            "Parse Error": "",
            "Test Available": "",
            "Timeout": "",
            "Locs": "",
            "Count": "",
            "Technique": "",
            "Cost": "",
            "GM Score": "",
            "Percentage Repaired": "",
            "Correct Locs": "",
            "Incorrect Locs": "",
            "Old incorrect Locs": "",
            "Correct Exprs": "",
            "Incorrect Exprs": "",
            "Old Incorrect Exprs": "",
            "Repairs": "",
            "First Output": "",
            "Second Output": "",
            "Error Output": ""
        }

    def new(self, index):
        self.data[index] = {}
        return index
    
    def add(self, index, key, data):
        self.data[index][key] = data

    def save(self, file):
        with open(file, "w") as f:
            json.dump(self.data, f)
    

def run_clara(directory, start, end):
    with open(pjoin(directory, 'combinations.json'), 'r') as f:
        combinations = json.loads(f.read())
    print(len(combinations))
    print(start)
    print(end)
    combinations = combinations[start:end]
    for c in tqdm(combinations):
        problem_name = c['problem_name']
        cfile = f'{c["correct_file"]}_solution.py'
        cnum = c["correct_num"]
        ifile = f'{c["incorrect_file"]}_solution.py'
        inum = c["incorrect_num"]
        g = c["g"]
        testcase_folder = pjoin(directory, 'ScrapeData', problem_name, 'testcases')
        outfolder = pjoin(directory, 'clara_raw_output', problem_name)
        if (not os.path.exists(outfolder)):
            os.makedirs(outfolder)
        outfile = open(pjoin(outfolder, f'{inum}_{cnum}_{str(g)}.txt'), 'w')
        errfile = open(pjoin(outfolder, f'{inum}_{cnum}_{str(g)}_err.txt'), 'w')
        if g == 0:
            clara_call = run_clara_repair(
                correct_file_path=cfile,
                incorrect_file_path=ifile,
                testcase=testcase_folder,
                out_file=outfile,
                error_file=errfile
            )
        else:
            clara_call = run_clara_graph(
                correct_file_path=cfile,
                incorrect_file_path=ifile,
                testcase=testcase_folder,
                out_file=outfile,
                error_file=errfile,
                g=g
            )
        outfile.close()
        errfile.close()
        if clara_call is not None:
            outfile = open(pjoin(outfolder, f'{inum}_{cnum}_{str(g)}.txt'), 'a')
            outfile.write(f'\nExitcode: {clara_call.returncode}')
            outfile.close() 

def run_clara_repair(correct_file_path, incorrect_file_path, testcase, out_file, error_file):
    '''
    Run clara repair
    '''
    command = f'clara repair {correct_file_path} {incorrect_file_path} --argsfile {testcase} --checkAllRep 1 --verbose 1'
    try:
        clara_call = subprocess.run([command], stdout=out_file, stderr=error_file, shell=True, timeout=300)
    except:
        clara_call = None
        pass
    return clara_call

def run_clara_graph(correct_file_path, incorrect_file_path, testcase, out_file, error_file, g):
    '''
    Run clara graph
    '''
    command = f'clara graph {correct_file_path} {incorrect_file_path} --argsfile {testcase} --checkAllRep 1 --verbose 1 --matchOp {str(g)}'
    try:
        clara_call = subprocess.run([command], stdout=out_file, stderr=error_file, shell=True, timeout=300)
    except:
        clara_call = None
        pass
    return clara_call

def get_combinations(directory, fn):
    '''
    Get all combinations of correct and incorrect as the following array of dictionary:
    {
        "problem_name": "",
        "correct_file": "",
        correct_num,
        "incorrect_file": "",
        incorrect_num,
        "g": ""
    }
    '''
    list_of_problems = os.listdir(pjoin(directory, 'ScrapeData'))
    graph_matching_options = [0, 1, 3]
    combinations = []
    for problem_name in list_of_problems:
        if problem_name not in ['ProblemList.txt', 'SolutionLists']:
            testcase_folder = pjoin(directory, 'ScrapeData', problem_name, 'testcases')
            correct_path = pjoin(directory, 'ScrapeData', problem_name, 'OK', 'python.3')
            incorrect_path = pjoin(directory, 'ScrapeData', problem_name, 'REJECTED', 'python.3')
            if os.path.exists(correct_path) and os.path.exists(incorrect_path):
                list_of_correct = get_problem_nums(correct_path)
                list_of_incorrect = get_problem_nums(incorrect_path)
                for incorrect_num in list_of_incorrect:
                    for correct_num in list_of_correct:
                        ifile = pjoin(incorrect_path, incorrect_num)
                        cfile = pjoin(correct_path, correct_num)
                        for g in graph_matching_options:
                            combinations.append(
                                {
                                    "problem_name": problem_name,
                                    "correct_file": cfile,
                                    "correct_num": correct_num,
                                    "incorrect_file": ifile,
                                    "incorrect_num": incorrect_num,
                                    "g": g
                                }
                            )
    print(f'Created combinations with length {len(combinations)}')
    with open(pjoin(directory, 'combinations.json'), 'w') as out:
        out.write(json.dumps(combinations, indent=4))


                # print(f'For Problem {problem_name}:')
                # print(f'    Total correct problems: {len(list_of_correct)}')
                # print(f'    Total incorrect problems: {len(list_of_incorrect)}')
                # print(f'    Running function: {fn}')

def get_problem_nums(path):
    correct = []
    if os.path.exists(path):
        for c in os.listdir(path):
            if ('solution.txt' in c):
                correct.append(c.split('_')[0])
                remove_unicode(pjoin(path, c))
        return correct

def parse_clara_output(directory, start=None, end=None):
    '''
    Parse the raw clara output from the files
    '''
    # All text based captures
    rep_correct = 'All Repaired'
    rep_incorrect = 'Old Repairs were incomplete, apply these repairs too:'
    rep_error = 'There are issues with the suggested repairs. The new program may not run.'
    rep_not_needed = 'No repair!'
    rep_partial = 'Partial Repaired'
    graph_matching_options = [0, 1, 3]
    timeout = 'TIMEOUT'
    loc_add = 'Locs Added:'
    loc_del = 'Locs Deleted:'
    loc_same = 'Locs Same'
    test_available = 'Test Case Available'
    test_not_available = 'Test Case Not Available'
    corr_locs = 'Locs in Correct Program Model'
    exit_code_text = 'Exitcode:'
    corr_exp = 'Exprs in Correct Program Model'
    incorr_locs = 'Locs in Incorrect Program Model'
    incorr_exps = 'Exprs in Incorrect Program Model'
    old_incorr_locs = 'Locs in Old Incorrect Program Model'
    old_incorr_exps = 'Exprs in Old Incorrect Program Model'
    num_Reps = 'Number of Repairs '

    
    with open(pjoin(directory, 'combinations.json'), 'r') as f:
        combinations = json.loads(f.read())
    if start is not None and end is not None:
        combinations = combinations[start:end]
    for c in tqdm(combinations):
        problem_name = c['problem_name']
        cfile = f'{c["correct_file"]}_solution.py'
        cnum = c["correct_num"]
        ifile = f'{c["incorrect_file"]}_solution.py'
        inum = c["incorrect_num"]
        g = c["g"]
        outfolder = pjoin(directory, 'clara_raw_output', problem_name)
        output_file = pjoin(outfolder, f'{ifile}_{cfile}_{str(g)}.txt')
        error_file = pjoin(outfolder, f'{ifile}_{cfile}_{str(g)}_err.txt')
        if not (os.path.exists(output_file) and os.path.exists(error_file)):
            continue
        

        
        
def parse_output(problem, correct, problems, correct_path, incorrect_path, graph_matching_options, testcase, rerun=False):
    outfolder = f'{data_dir}/batch_tests_output/{problem}/'
    for ifile in tqdm(problems, desc="incorrect", position=2, leave=False):
        try:
            results = ClaraResults()
            # for cfile in tqdm(correct, desc="correct", position=3, leave=False):
            for cfile in correct:
                cdired = correct_path + cfile + '_solution.py'
                idired = incorrect_path + ifile + '_solution.py'
                for g in graph_matching_options:
                    if not (os.path.exists(f'{outfolder}{ifile}_{cfile}_{str(g)}.txt') and os.path.exists(f'{outfolder}{ifile}_{cfile}_{str(g)}_err.txt')):
                        continue
                    idx = results.new(f'{cfile}_{str(g)}')
                    output = open(f'{outfolder}{ifile}_{cfile}_{str(g)}.txt', 'r').read()
                    err = open(f'{outfolder}{ifile}_{cfile}_{str(g)}_err.txt', 'r').read()
                    results.add(idx,"First Output", output)
                    # err = clara_call.stderr.decode('utf-8')
                    results.add(idx,"Error Output", err)
                    formatted_output = output.split('\n')
                    temp = list(
                        filter(lambda x: exit_code_text in x, formatted_output))
                    temp = temp[0].split(exit_code_text)[-1].strip()
                    exitcode = temp
                    if ((g == 1 or g == 3) and 'SCORE TOO LESS' in output):
                        continue
                    results.add(idx,'Correct File', cfile)
                    results.add(idx,'Incorrect File', ifile)
                    if (test_available in output):
                        results.add(idx,'Test Available', 'Yes')
                    elif (test_not_available in output):
                        results.add(idx,'Test Available', 'No')

                    # Locs + Exp
                    temp = list(
                        filter(lambda x: corr_locs in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(corr_locs)[-1].strip()
                        results.add(idx,"Correct Locs", temp)
                    temp = list(
                        filter(lambda x: incorr_locs in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(incorr_locs)[-1].strip()
                        results.add(idx,"Incorrect Locs", temp)
                    temp = list(
                        filter(lambda x: old_incorr_locs in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(old_incorr_locs)[-1].strip()
                        results.add(idx,"Old incorrect Locs", temp)
                    temp = list(
                        filter(lambda x: corr_exp in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(corr_exp)[-1].strip()
                        results.add(idx,"Correct Exprs", temp)
                    temp = list(
                        filter(lambda x: incorr_exps in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(incorr_exps)[-1].strip()
                        results.add(idx, "Incorrect Exprs", temp)
                    temp = list(
                        filter(lambda x: old_incorr_exps in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(old_incorr_exps)[-1].strip()
                        results.add(idx,"Old Incorrect Exprs", temp)
                    temp = list(
                        filter(lambda x: num_Reps in x, formatted_output))
                    if len(temp):
                        temp = temp[0].split(num_Reps)[-1].strip()
                        results.add(idx, "Repairs", temp)

                    results.add(idx,"Technique", g)
                    if g == 0:
                        results.add(idx,"Locs", 0)
                        results.add(idx,"Count", 0)
                    else:
                        temp = list(
                            filter(lambda x: 'Score:' in x, formatted_output))
                        if len(temp):
                            temp = temp[0].split("Score:")[-1].strip()
                            results.add(idx,"GM Score", temp)
                        if loc_add in output:
                            results.add(idx,"Locs", 'Add')
                            temp = list(
                                filter(lambda x: loc_add in x, formatted_output))
                            temp = temp[0].split(loc_add)[-1].strip()
                            results.add(idx,"Count", temp)
                        elif loc_same in output:
                            results.add(idx,"Locs", 'Same')
                        elif loc_del in output:
                            results.add(idx,"Locs", 'Del')
                            temp = list(
                                filter(lambda x: loc_del in x, formatted_output))
                            temp = temp[0].split(loc_del)[-1].strip()
                            results.add(idx,"Count", temp)
                    if (timeout in output):
                        results.add(idx,"Timeout", 'Yes')

                    if (exitcode == 0):
                        results.add(idx,"Repair", 'Yes')
                        results.add(idx,"Structure Mismatch", 'False')
                        results.add(idx,"Parse Error", 'No')
                        if (rep_correct in output):
                            results.add(idx,"Repair Correct", 'Yes')
                        elif (rep_partial in output):
                            results.add(idx,"Repair Correct", 'Partial')
                        elif (rep_not_needed in output):
                            results.add(idx,"Repair Correct", 'Not Needed')
                        elif (rep_error in output):
                            results.add(idx,"Repair Correct", 'Error')
                        elif (rep_incorrect in output):
                            results.add(idx,"Repair Correct", 'No')
                        temp = list(
                            filter(lambda x: 'Cost:' in x, formatted_output))
                        if len(temp):
                            temp = temp[0].split("Cost:")[-1].strip()
                            results.add(idx,"Cost", temp)
                        temp = list(
                            filter(lambda x: 'Percentage of the model modified' in x, formatted_output))
                        if len(temp):
                            temp = temp[0].split(
                                'Percentage of the model modified')[-1].strip()
                            results.add(idx,"Percentage Repaired", temp)
                    else:
                        if ('StructMismatch' in err):
                            results.add(idx,"Structure Mismatch", 'True')
                        else:
                            results.add(idx,"Structure Mismatch", 'False')
                        if (timeout in err or 'Timeout' in err):
                            results.add(idx,"Timeout", 'Yes')
                        results.add(idx,"Repair", 'No')
                        results.add(idx,"Repair Correct", 'Error')
                        if ("Parse Error!" in output):
                            results.add(idx,"Parse Error", 'Yes')
                        else:
                            results.add(idx,"Parse Error", 'No')
                        results.add(idx,"Cost", 0)
                    if (rerun):
                        clara_call_match = subprocess.run(['clara match ' + cdired + ' ' + idired + ' --argsfile ' + testcase],
                                                        stdout=subprocess.PIPE,
                                                        shell=True)
                        output_match = clara_call_match.stdout.decode('utf-8')
                        exitcode_match = clara_call_match.returncode
                        results.add(idx,"Second Output", output_match)
                        if (exitcode_match == 0):
                            if ('No match!' in output_match):
                                results.add(idx,"Match", 'No')
                            else:
                                results.add(idx,"Match", 'Yes')
                        else:
                            results.add(idx,"Match", 'Error')
                    else:
                        results.add(idx, "Second Output", "No second run!")
                        results.add(idx, "Match", "No Output")

            incorrect_file_no = ifile.split('_')[0]
            if (not os.path.exists(f'{data_dir}/parsed_results/{problem}/')):
                os.makedirs(f'{data_dir}/parsed_results/{problem}/')
            results.save(f'{data_dir}/parsed_results/{problem}/{incorrect_file_no}_{str(g)}.json')
        except Exception as e:
            print(e)


def batch_run_json( problem, 
                    correct, 
                    problems, 
                    correct_path, 
                    incorrect_path, 
                    graph_matching_options, 
                    testcase):
    print(problems)
    for ifile in tqdm(problems, desc="incorrect", position=0):
        # incorrect file
        # ifile = problems[x]
        outfolder = f'{data_dir}/batch_tests_output/{problem}/'
        if (not os.path.exists(outfolder)):
            os.makedirs(outfolder)

        for cfile in correct:
            cdired = correct_path + cfile + '_solution.py'
            idired = incorrect_path + ifile + '_solution.py'
            # go through each graph matching options
            for g in graph_matching_options:
                if os.path.exists(f'{outfolder}{ifile}_{cfile}_{str(g)}.txt') or os.path.exists(f'{outfolder}{ifile}_{cfile}_{str(g)}_err.txt'):
                    continue
                else:
                    outfile = open(f'{outfolder}{ifile}_{cfile}_{str(g)}.txt', "w")
                    outfile_err = open(f'{outfolder}{ifile}_{cfile}_{str(g)}_err.txt', 'w')
                    if g == 0:
                        clara_call = subprocess.run(['clara repair ' + cdired + ' ' + idired + ' --argsfile ' + testcase + ' --checkAllRep 1 --verbose 1'],
                                                    stdout=outfile, stderr=outfile_err,
                                                    shell=True)
                    else:
                        clara_call = subprocess.run(['clara graph ' + cdired + ' ' + idired + ' --argsfile ' + testcase + ' --checkAllRep 1 --verbose 1 --matchOp ' + str(g)],
                                                    stdout=outfile, stderr=outfile_err,
                                                    shell=True)
                    outfile.close()
                    outfile_err.close()
                    outfile = open(f'{outfolder}{ifile}_{cfile}_{str(g)}.txt', "a")
                    outfile.write(f'\nExitcode: {clara_call.returncode}')
                    outfile.close() 

def thread_run(lst, fn='parse_output'):
    # logging.info("Thread %s: starting", thread)
    for problem_name in tqdm(lst, position=1, desc='problems', leave=False):
        if problem_name not in ['ProblemList.txt', 'SolutionLists']:
            # problem_name = '1A'
            testcase = f'{data_dir}/ScrapeData/{problem_name}/testcases/'

            correct_path = path + problem_name + '/OK/python.3/'
            incorrect_path = path + problem_name + '/REJECTED/python.3/'
            if os.path.exists(correct_path) and os.path.exists(incorrect_path): 
                correct = get_problem_nums(correct_path)
                probs = get_problem_nums(incorrect_path)
                size = len(probs)

                print(f'Total correct problems: {len(correct)}')
                print(f'Total incorrect problems: {size}')
                print(f'Running function: {fn}')
                if fn=='batch_run_json':
                    batch_run_json(problem_name, correct, probs, correct_path, incorrect_path, graph_matching_options, testcase)
                elif fn=='parse_output':
                    parse_output(problem_name, correct, probs, correct_path, incorrect_path, graph_matching_options, testcase)


# /home/mac9908/clara/problems/Output
if __name__=='__main__':
    directory = sys.argv[1]
    fn = sys.argv[2]
    if fn == 'get_combinations':
        get_combinations(directory, fn)
    elif fn == 'clara':
        start = int(sys.argv[3])
        end = int(sys.argv[4])
        run_clara(directory=directory, start=start, end=end)
