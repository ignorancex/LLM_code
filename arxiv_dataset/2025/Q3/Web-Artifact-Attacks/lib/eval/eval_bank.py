import torch
import numpy as np 
import torchmetrics
import itertools

class EvalBank(): 


	def eval_label(self, output, pair, unique_labels): 
		logits = output["model_output"]
		labels = output["label"]
		pred_concept_per_race_gender = {x:[0, 0] for x in unique_labels}

		for per_image_score, label, in zip(logits, labels):
			if pair[torch.argmax(per_image_score).item()] == label: 
				pred_concept_per_race_gender[label][0] += 1
			
			pred_concept_per_race_gender[label][1] += 1

		accs_all = []
		to_write = ["pair,accuracy\n"]
		for id, value in pred_concept_per_race_gender.items(): 
			acc = value[0] / value[1]
			accs_all.append(acc)
			to_write.append(f"{id},{acc}\n")
		
		to_write.append(f"Average, {sum(accs_all) / len(accs_all)}")
		return to_write