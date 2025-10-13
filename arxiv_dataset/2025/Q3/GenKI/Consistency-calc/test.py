from ctrleval import CTRLEval
task = 'senti' # evaluation for sentiment-controlled text generation
scorer = CTRLEval(iwf_dir='iwf_full.txt', prompt_dir='./prompt/prompt_{}.txt'.format(task), verbal_dir='./prompt/verbal_{}.txt'.format(task), model_name_or_path='transformers/pegasus-large')
data = ['The book is about NLP. It depicts fancy models.']
prefix = ['The book']
label = ['positive']
a=scorer.score(aspect='coh', data=data, batch_size=2) # evaluation of coherence
print(a)
a=scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=2) # evaluation of consistency
print(a)
a=scorer.score(aspect='ar', data=data, label=label, batch_size=2) # evaluation of attribute relevance
print(a)