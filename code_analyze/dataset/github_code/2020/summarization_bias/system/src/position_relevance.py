# Position relevance segment

import re
from nltk.util import ngrams

source_topic = 'sample_data/source.txt'
reference = 'sample_data/ref_summary.txt'

source_topic = open(source_topic,'r').read()
reference = open(reference,'r').read()

s = source_topic
s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 5))

def gram_maker(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, 2))
    return output

def chunk(in_string,num_chunks):
    chunk_size = len(in_string)//num_chunks
    if len(in_string) % num_chunks: chunk_size += 1
    iterator = iter(in_string)
    for _ in range(num_chunks):
        accumulator = list()
        for _ in range(chunk_size):
            try: accumulator.append(next(iterator))
            except StopIteration: break
        yield ''.join(accumulator)

# Counting ngrams presence in various segments.
one = 0
two = 0
three = 0
# ref = 
temp = ''
counter_tot = 0
# for s in range(len(source_topic)):
a,b,c = list(chunk(source_topic,3))
grams = gram_maker(reference)
for elem in grams:
    if ' '.join(list(elem)) in a:
        one += 1
    elif ' '.join(list(elem)) in b:
        two += 1
    elif ' '.join(list(elem)) in c:
        three += 1
counter_tot += len(grams)

print('Content from first segment',one/counter_tot)
print('Content from second segment',two/counter_tot)
print('Content from third segment',three/counter_tot)