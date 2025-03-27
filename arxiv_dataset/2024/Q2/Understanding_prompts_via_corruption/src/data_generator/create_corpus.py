from datasets import load_dataset
import random

# create corpus.txt
dataset = load_dataset("c4", 'en', split='train', streaming=True)
random.seed(42)
shuffled_dataset = dataset.shuffle(buffer_size=100000, seed=42)


random_sent_from_c4 = []
n = 200
i=0
with open('data-generator/data-for-corruptions/corpus.txt', 'w') as f:
    for item in shuffled_dataset:
        text = item['text']
        text = text.split('.')
        text = list(filter(None, text))
        random_sent = random.choice(text)
        random_sent = random_sent.strip()
        random_sent = random_sent.strip('\n')
        random_sent = random_sent.replace('\n', " ")
        random_sent = random_sent.strip('"')
        random_sent = random_sent.lstrip()
        if random_sent.strip() != "" or len(random_sent) >= 8:
            f.write(random_sent)
            f.write('\n')
            i+=1
        if i==n:
            break

# few lines were manually cleaned (e.g. single words)