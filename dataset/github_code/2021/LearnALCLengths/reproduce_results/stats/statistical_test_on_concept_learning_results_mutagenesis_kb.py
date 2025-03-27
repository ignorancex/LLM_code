import sys, os

base_path = os.path.dirname(os.path.realpath(__file__)).split('reproduce_results')[0]
sys.path.append(base_path)

from helper_functions import wilcoxon_statistical_test
import json
import warnings

warnings.filterwarnings('ignore')
path = base_path+"Datasets/mutagenesis/Results"

print()
print('#'*50)
print('On {} KB'.format(path.split('/')[-2].upper()))
print('#'*50)

with open(path+"/concept_learning_results_celoe_clp.json") as results_clp:
    clp_data = json.load(results_clp)

with open(path+"/concept_learning_results_celoe.json") as results_celoe:
    celoe_data = json.load(results_celoe)
    
with open(path+"/concept_learning_results_ocel.json") as results_ocel:
    ocel_data = json.load(results_ocel)
    
with open(path+"/concept_learning_results_eltl.json") as results_eltl:
    eltl_data = json.load(results_eltl)

valid_attributes = [attribute for attribute in celoe_data if attribute not in ['Prediction', 'Learned Concept']]
for algo, algo_data in zip(['CELOE', 'OCEL', 'ELTL'], [celoe_data, ocel_data, eltl_data]):
    statistical_test_results_dict = {attr: dict() for attr in valid_attributes}
    print()
    print('*'*40)
    print('Statistics CELOE-CLP vs {}'.format(algo))
    print('*'*40)
    for attribute in valid_attributes:
        print("Test on "+attribute+":")
        data1 = algo_data[attribute]
        data2 = clp_data[attribute]
        stats, p = wilcoxon_statistical_test(data1, data2)
        print()
        statistical_test_results_dict[attribute].update({"p-value": p, "stats": stats})
    with open(path+"/wilcoxon_statistical_test_CELOE-CLP_vs_{}.json".format(algo), "w") as stat_test:
        json.dump(statistical_test_results_dict, stat_test, indent=3)

    print("\nStatistical test results saved in "+path+"/")