import sys
# import nltk
# nltk.find('/Datasets/nltk_data-gh-pages/nltk_data')
# Add the directory of promptbench to the Python path
sys.path.append('/root/code/MLLMGuard/promptbench')

# Now you can import promptbench by name
import promptbench as pb
from promptbench.models import LLMModel
from promptbench.prompt_attack import Attack


# create model

# create dataset

# try part of the dataset
dataset = dataset[:10]

# create prompt
prompt = "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: \nQuestion: {content}\nAnswer:"

# define the projection function required by the output process
def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

# define the evaluation function required by the attack
# if the prompt does not require any dataset, for example, "write a poem", you still need to include the dataset parameter
def eval_func(prompt, dataset, model):
    preds = []
    labels = []
    for d in dataset:
        input_text = pb.InputProcess.basic_format(prompt, d)
        raw_output = model(input_text)

        output = pb.OutputProcess.cls(raw_output, proj_func)
        preds.append(output)

        labels.append(d["label"])
    
    return pb.Eval.compute_cls_accuracy(preds, labels)
    
# define the unmodifiable words in the prompt
# for example, the labels "positive" and "negative" are unmodifiable, and "content" is modifiable because it is a placeholder
# if your labels are enclosed with '', you need to add \' to the unmodifiable words (due to one feature of textattack)
unmodifiable_words = ["positive\'", "negative\'", "content"]

# print all supported attacks
print(Attack.attack_list())

# create attack, specify the model, dataset, prompt, evaluation function, and unmodifiable words
# verbose=True means that the attack will print the intermediate results
attack = Attack(model_t5, "stresstest", dataset, prompt, eval_func, unmodifiable_words, verbose=True)

# print attack result
print(attack.attack())