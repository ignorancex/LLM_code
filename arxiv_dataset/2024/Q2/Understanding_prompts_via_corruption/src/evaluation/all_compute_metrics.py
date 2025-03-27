import os

modelname = "30.33B_opt-30b"
set = "set100"

print(modelname)
print(set)

print("############ "+ "empty" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_empty.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "only_instruction" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_only_instruction.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "only_demostration" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_only_demostration.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_n_demo" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_n_demo.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "all_instr_n_demo" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_all_instr_n_demo.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_randomwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_randomwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_frequentwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_frequentwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_50per_replaced_frequentwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_50per_replaced_frequentwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_add_50percent_frequqentwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_add_50percent_frequqentwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_add_100percent_frequqentwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_add_100percent_frequqentwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_add_at_end_frequentwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_add_at_end_frequentwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "wrong_labels_labelspace" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_wrong_labels_labelspace.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "wrong_labels_labelspace_halfcases" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_wrong_labels_labelspace_halfcases.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "labels_randomwords" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_labels_randomwords.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "label_empty" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_label_empty.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "input_empty" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_input_empty.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "instr_after_demo" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_instr_after_demo.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))

print("############ "+ "input_ood" + "  #############")
os.system('python evaluation/compute_metrics.py --predictions output/{}/{}/predicted_examples_input_ood.jsonl --track default --compute_per_category_metrics --compute_per_task_metrics'.format(modelname, set))
