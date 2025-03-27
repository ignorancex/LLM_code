import json, time
import numpy as np
from collections import defaultdict
import os, sys, random
import argparse

currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("other_learning_systems")[0])

from ontolearn.binders import DLLearnerBinder
from concept_length_metric import concept_length

from ontolearn import KnowledgeBase

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_systems', nargs='+', type=str, default=['celoe', 'eltl', 'ocel'], help="DL-Learner concept learners to use, should be one of celoe, eltl, ocel")
    parser.add_argument('--knowledge_bases', nargs='+', type=str, default=['semantic_bible', 'carcinogenesis', 'mutagenesis', 'vicodi'], \
                        help="List of knowledge bases on which to learn class expressions, should be strings separated with white space and should belong in semantic_bible, carcinogenesis, mutagenesis, vicodi")
    parser.add_argument('--max_num_lp', type=int, default=100, help="The maximum number of learning problems to solve on each knowledge base")
    parser.add_argument('--max_exec_time', type=int, default=120, help="The maximum execution time per algorithm")
    args = parser.parse_args()
        
    # Fix a random seed for sampling learning problems, not that we may not want to solve all problems
    random.seed(1)
    
    # Path to the DL-Learner package
    dl_learner_binary_path = currentpath.split("other_learning_systems")[0]+'dllearner-1.4.0/'
    
    for kb in args.knowledge_bases:
        kb_path = currentpath.split("other_learning_systems")[0]+'Datasets/'+kb+'/'+kb+'.owl'
        if kb == 'family-benchmark':
            kb_path = currentpath.split("other_learning_systems")[0]+'Datasets/'+kb+'/'+kb+'_rich_background.owl'
        knowledge_base = KnowledgeBase(path=kb_path)
        with open(currentpath.split("other_learning_systems")[0]+'Datasets/'+kb+'/Learning_problems/learning_problems.json') as json_file:
            settings = json.load(json_file)
        
        kb_namespace = list(knowledge_base.individuals())[0].get_iri().get_namespace()
        kb_prefix = kb_namespace[:kb_namespace.rfind("/")+1]
        
        learning_probs = list(settings.items())
        random.shuffle(learning_probs)
        max_num_lp = args.max_num_lp
        
        for model in args.learning_systems:
            algo = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model=model)
            Result_dict = {'F-measure': [], 'Accuracy': [], 'Runtime': [], 'Prediction': [], 'Length': [], 'Learned Concept': []}
            Avg_result = defaultdict(lambda: defaultdict(float))
            iterator = 0
            print("#"*60)
            print("{} running on {} knowledge base".format(model.upper(), kb_path.split("/")[-2]))
            print("#"*60)
            for str_target_concept, examples in learning_probs:
                print('TARGET CONCEPT:', str_target_concept)
                iterator += 1
                p = [kb_prefix+ind for ind in examples['positive examples']]
                n = [kb_prefix+ind for ind in examples['negative examples']]
    
                t0 = time.time()
                best_pred_algo = algo.fit(pos=p, neg=n, max_runtime=args.max_exec_time).best_hypotheses() # Start learning
                duration = time.time()-t0
                
                print('Best prediction: ', best_pred_algo)
                print()
                if model == 'ocel': # No F-measure for OCEL
                    Result_dict['F-measure'].append(-1.)
                else:
                    Result_dict['F-measure'].append(best_pred_algo['F-measure']/100)
                Result_dict['Accuracy'].append(best_pred_algo['Accuracy']/100)
                if not 'Runtime' in best_pred_algo or best_pred_algo['Runtime'] is None:
                    Result_dict['Runtime'].append(duration)
                else:
                    Result_dict['Runtime'].append(best_pred_algo['Runtime'])
                if best_pred_algo['Prediction'] is None:
                    Result_dict['Prediction'].append('None')
                    Result_dict['Length'].append(15)
                else:
                    Result_dict['Prediction'].append(best_pred_algo['Prediction'])
                    Result_dict['Length'].append(concept_length(best_pred_algo['Prediction']))
                Result_dict['Learned Concept'].append(str_target_concept)

                if iterator == max_num_lp: break

            for key in Result_dict:
                if key in ['Prediction', 'Learned Concept']: continue
                Avg_result[key]['mean'] = np.mean(Result_dict[key])
                Avg_result[key]['std'] = np.std(Result_dict[key])

            with open(currentpath.split("other_learning_systems")[0]+'Datasets/'+kb+'/Results/concept_learning_results_'+model+'.json', 'w') as file_descriptor:
                        json.dump(Result_dict, file_descriptor, ensure_ascii=False, indent=3)

            with open(currentpath.split("other_learning_systems")[0]+'Datasets/'+kb+'/Results/concept_learning_avg_results_'+model+'.json', 'w') as file_descriptor:
                        json.dump(Avg_result, file_descriptor, indent=3)

            print("Avg results for {}: {}".format(model.upper(), Avg_result))
            print()
