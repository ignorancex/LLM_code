import nest_asyncio
import argparse
import json
import os

from ragas.metrics import faithfulness, answer_correctness, context_relevancy
from datasets import Dataset 
from ragas import evaluate
import evaluate as hf_evaluate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_model', type=str, default='Anita-3')
	parser.add_argument('--llm_retriever', type=str, default='openAI')
	
	args = parser.parse_args()
	model_name = args.llm_model
	retriever_name = args.llm_retriever
	
	path = 'unipa-gpt/' # path till main folder
	
	with open(path+'results/qa_'+model_name+'.json', 'r') as f:
		qa_preds = json.load(f)
	
	with open(path+'corpora/qa_golden.json', 'r') as f:
		qa_golden = json.load(f)
	
	with open(path+'results/qa_retrieved_'+retriever_name+'.json', 'r') as f:
		qa_context = json.load(f)
	
	os.environ["OPENAI_API_KEY"] = ""
	# insert your openAI key

	bleu = hf_evaluate.load("bleu")
	rouge = hf_evaluate.load('rouge')
	
	nest_asyncio.apply()
	scores = []
	
	for idx in range(len(qa_preds)):
	
		print(f'Sentence {idx+1} evaluation')
		
		data_samples = {
		  'question': [qa_golden[idx]['question']],
		  'answer': [qa_preds[idx][2]],
		  'contexts' : [qa_context[idx]],
		  'ground_truth': [qa_golden[idx]['golden']]
		}
		
		score = evaluate(dataset, metrics=[faithfulness, answer_correctness, context_relevancy], raise_exceptions=False)
		
		bleu_score = (bleu.compute(predictions=[qa_preds[idx][2]], references=[qa_golden[idx]['golden']]))["bleu"]
		rouge_score = rouge.compute(predictions=[qa_preds[idx][2]], references=[qa_golden[idx]['golden']])
		
		scores_dict = {
			'QA': idx+1,
			'model': model_name,
			'faithfulness': score['faithfulness'],
			'answer_correctness': score['answer_correctness'],
			'context_relevancy' : score['context_relevancy'],
			'bleu' : bleu_score,
			'rouge' : rouge_score
		}
		
		scores.append(scores_dict)
	
	with open(path+'qa_metrics.json', 'w') as f:
		json.dump(scores, f, indent=4)
