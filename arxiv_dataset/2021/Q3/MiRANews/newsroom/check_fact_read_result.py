#encoding=utf-8

import os
import sys
import time
import json
import random

if __name__ == '__main__':
    coverage_scores = []
    coverage_other_scores = []
    coverage_cluster_scores = []
    win_over_num = 0
    example_total_num = 0
    win_fact_num = 0
    fact_total_num = 0
    cluster_total_num = 0
    for i in range(0, 4):
        with open('tmp_'+str(i)+'.checkpoint') as f:
            line = f.read().strip()
            json_obj = json.loads(line)
            win_over = json_obj['win_over_num']
            example_num = json_obj['example_total_num']
            win_over_fact = json_obj['win_fact_num']
            fact_num = json_obj['fact_total_num']
            coverage_score = json_obj['coverage_scores']
            coverage_other_score = json_obj['coverage_other_scores']
            coverage_cluster_score = json_obj['coverage_cluster_scores']
            cluster_num = json_obj['finished_exmples']

            cluster_total_num += cluster_num
            win_over_num += win_over
            example_total_num += example_num
            win_fact_num += win_over_fact
            fact_total_num += fact_num
            coverage_scores.extend(coverage_score)
            coverage_other_scores.extend(coverage_other_score)
            coverage_cluster_scores.extend(coverage_cluster_score)
    print ('number of clusters:', cluster_total_num)
    print ('number of examples:', example_total_num)
    print ('number of facts:', fact_total_num)
    print ('win_over_rate(summaries):', win_over_num/example_total_num)
    print ('win_over_rate(facts):', win_fact_num/fact_total_num)
    print ('avg coverage_scores:', sum(coverage_scores)/len(coverage_scores))
    print ('avg coverage_scores with other docs:', sum(coverage_other_scores)/len(coverage_other_scores))
    print ('avg coverage_scores with docs in the cluster:', sum(coverage_cluster_scores)/len(coverage_cluster_scores))

