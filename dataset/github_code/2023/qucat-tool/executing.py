import os
import numpy as np
from generating import generate_test_suite
from GenerateUnitTest import GenerateUnitTest
import time


import math


from qiskit import (
    # IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.tools.visualization import circuit_drawer
from qiskit import QuantumCircuit, assemble
import importlib
from rpy2 import robjects as robjects



def execute_quantum_program(inputID, outputID, num_qubit, input, module_name, times):  # module_name is the name of the quantum program file
    q = QuantumRegister(num_qubit)
    c = ClassicalRegister(len(outputID))
    qc = QuantumCircuit(q, c)
    for j in range(len(inputID)):
        if input[len(inputID) - 1 - j] == '1':
            qc.x(inputID[j])
    module = importlib.import_module(module_name)
    run_method = getattr(module, "run")
    run_method(qc)
    result = execute(qc, Aer.get_backend('qasm_simulator'), shots=times).result().get_counts(qc)
    return result


def execute_test_case(input, config_dic):
    valid_input = config_dic['valid_input']
    valid_output = config_dic['valid_output']
    pt = config_dic['p']
    right_output = []
    p = []
    count_times = 0
    fre = []
    test_result = 'PASS'
    flag_wrong = False

    for j in range(len(valid_input)):
        if valid_input[j] == input and pt[j] > 0:
            right_output.append(valid_output[j])
            p.append(pt[j])
            count_times += 1

    #!!!!!
    result = execute_quantum_program(config_dic['inputID'], config_dic['outputID'], config_dic['num_qubit'], input, config_dic['module_name'], count_times*100)

    # judge wrong outputs
    for key in list(result.keys()):
        if key not in right_output:
            flag_wrong = True
    #
    # chi test
    if flag_wrong == False:  # no wrong output
         if count_times != 1:
            for o in right_output:
                if o in result:
                    fre.append(result[o])
                else:
                    fre.append(0)
            p = np.array(p)
            fre = np.array(fre)
            p = robjects.FloatVector(p)
            fre = robjects.FloatVector(fre)
            robjects.r('''
                chitest<-function(observed,theoretical){
                    test_result <- chisq.test(x = observed,p = theoretical)
                    pvalue = test_result$p.value
                    return (pvalue)
                }''')
            pvalue = robjects.r['chitest'](fre, p)

            if pvalue[0] < config_dic['significance_level']:
                test_result = 'wodf'

    else:
        test_result = 'uof'

    return test_result, result, count_times*100


# doing testing
def comb_testing_scenario1(config_dic):
    result_root = os.path.join(config_dic['program_folder'], config_dic['module_name'], config_dic['module_name']+'_FunOne')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    generate_test_suite(config_dic, 'FunOne', config_dic['k'])
    test_suite_file = os.path.join(result_root, config_dic['module_name'] + '_input_k' + str(config_dic['k']) + '.txt')
    execution_time = 0
    with open(test_suite_file, 'r') as input_file:
        input_file.readline()
        input_str = input_file.readline().replace('\n','').replace('\t','')
        count_fail = 0
        list_data = []
        counts_list = []
        inputs_list = []
        inputs_str_list = []
        while input_str != "":
            start = time.time()
            test_result, result, counts = execute_test_case(input_str,config_dic)
            end = time.time()
            t = end-start
            execution_time += t
            counts_list.append(counts)
            list_data.append([input_str, result, test_result])
            inputs_list.append(int(input_str,2))
            inputs_str_list.append(input_str)
            input_str = input_file.readline().replace('\n', '').replace('\t', '')
            if test_result != 'PASS':
                count_fail += 1
    test_result_file = os.path.join(result_root, config_dic['module_name'] + '_result_k' + str(config_dic['k']) + '.txt')
    GenerateUnitTest.generateUnitTestClass(config_dic['module_name'], config_dic['inputID'], config_dic['outputID'], config_dic['num_qubit'], inputs_list, inputs_str_list, counts_list, config_dic['program_folder'], 1, config_dic['k'])
    with open(test_result_file, 'w') as result_file:
        result_file.write('inputs' + '\t' + 'outputs' + '\t' + 'result' +'\n')
        print("The number of test cases:" + str(len(list_data)))
        for i in range(len(list_data)):
            result_file.write(list_data[i][0] + '\t' + str(list_data[i][1]) + '\t' + list_data[i][2])
            result_file.write('\n')
    print("The number of failing test cases:" + str(count_fail))
    print("The execution time for running the quantum program is "+str(execution_time)+"s.")



        #
        # d = {'inputs','outputs', 'test results'}
        # df = pandas.DataFrame(list_data,columns=d)
        # df.to_excel(os.path.join(result_root, config_dic['module_name'] + '.xlsx'), sheet_name='test result', index=False)


def comb_testing_scenario2(config_dic):
    result_root = os.path.join(config_dic['program_folder'], config_dic['module_name'], config_dic['module_name']+'_FunTwo')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    #print("++++++configuration dic o ="+str(config_dic['o'])+"+++++++")
    flag_fail = False
    list_data = []
    counts_list = []
    inputs_list = []
    inputs_str_list = []
    execution_time = 0
    count_fail = 0
    for k in range(2, config_dic['k']+1):
        generate_test_suite(config_dic, 'FunTwo', k)
        test_suite_file = os.path.join(result_root, config_dic['module_name'] + '_input_k' + str(k) + '.txt')
        with open(test_suite_file, 'r') as input_file:
            input_file.readline()
            input_str = input_file.readline().replace('\n', '').replace('\t', '')
            while input_str != "":
                start = time.time()
                test_result, result, counts = execute_test_case(input_str, config_dic)
                end = time.time()
                t = end - start
                execution_time += t
                counts_list.append(counts)
                inputs_list.append(int(input_str,2))
                inputs_str_list.append(input_str)
                list_data.append([input_str, result, test_result])
                input_str = input_file.readline().replace('\n', '').replace('\t', '')
                count_fail += 1
                if test_result != 'PASS':
                    flag_fail = True
                    break
        if flag_fail == True:
            break
    test_result_file = os.path.join(result_root, config_dic['module_name'] + '_result' + '.txt')
    GenerateUnitTest.generateUnitTestClass(config_dic['module_name'], config_dic['inputID'], config_dic['outputID'], config_dic['num_qubit'], inputs_list, inputs_str_list, counts_list, config_dic['program_folder'], 2, config_dic['k'])
    with open(test_result_file, 'w') as result_file:
        result_file.write('inputs' + '\t' + 'outputs' + '\t' + 'result' + '\n')
        for i in range(len(list_data)):
            result_file.write(list_data[i][0] + '\t' + str(list_data[i][1]) + '\t' + list_data[i][2])
            result_file.write('\n')
        if flag_fail == False:
            result_file.write("QuCAT cannot kill the test.")
    if flag_fail == True:
        print("The number of test case that triggers the fault is " + str(count_fail) + '.')
    else:
        print("QuCAT cannot kill the test.")
    print("The execution time for running the quantum program is " + str(execution_time) + "s.")

if __name__ == '__main__':
    config_dic = {}
    config_dic['valid_input'] = ['00', '00', '01', '01','10','10','11','11']
    config_dic['valid_output'] = ['00', '11', '00', '11','01', '10', '01', '10']
    config_dic['p'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    config_dic['n'] = 2
    config_dic['k'] = 4
    config_dic['inputID'] = [0,1]
    config_dic['outputID'] = [0,1]
    config_dic['num_qubit'] = 2
    config_dic['module_name'] = 'entanglement'
    config_dic['significance_level'] = 0.01
    config_dic['pict_root'] = '.'
    config_dic['program_folder'] = '/Users/xinyi/Documents/NordIQuEst/quito-main/tutorial/example'

    comb_testing_scenario2(config_dic)




        #
        # input_file = open(config_dic[''])
        # dir_root = './inputs_' + str(K) + '/' + program + '/' + mutant + '_o' + str(o) + '/'
        # # input_file = open('./inputs/'+'input_'+program+"_"+mutant+".txt")
        # count_list = []  # the list of test cases in different test suites
        # fail_list = []  # the list of test cases that fail in different test suites
        # fail_suite_count = 0
        # for i in range(K):
        #     input_file = open(dir_root + 'input_' + program + '_' + mutant + '_' + str(i) + '.txt')
        #     count = 0  # the number of test cases in one test suite
        #     count_fail = 0  # the number of test cases that fail in one test suite
        #     input_file.readline()
        #     input_str = input_file.readline().replace('\n', '').replace('\t', '')
        #     while input_str != "":
        #         count += 1
        #         input = int(input_str, 2)
        #         flag_fail, f = calculate_fail_number_GA(input, mutant, 'Comb', program, o)
        #         input_str = input_file.readline().replace('\n', '').replace('\t', '')
        #         if flag_fail == True:
        #             count_fail += 1
        #     # f = open('./results/'+program+'/' + program + '_' + mutant + '_' + 'Comb' + '.txt', 'a')
        #     f.write('fail / total :' + str(count_fail) + ' / ' + str(count))
        #     f.write('\n')
        #     if count_fail > 0:
        #         fail_suite_count += 1
        #     fail_list.append(count_fail)
        #     count_list.append(count)
        # f.write('\n')
        # for i in range(K):
        #     f.write(str(fail_list[i]) + ' / ' + str(count_list[i]))
        #     f.write('\t')
        # f.write('\n')
        # f.write(
        #     'number of fail suites / total number of suites : ' + str(fail_suite_count) + '/' + str(len(count_list)))
        # f.write('\n')
        # f.write('number of fail cases / total number of cases : ' + str(sum(fail_list)) + ' / ' + str(sum(count_list)))
        # f.write('\n')
        # return count_list

