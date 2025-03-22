# Position jumble
import re
from nltk.util import ngrams

file_name = 'jumbled_position.txt'
def gram_maker(s, n_breaker):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, n_breaker))
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


def print_3(source_topic,reference):
	n_breaker = 2
    one = 0
    two = 0
    three = 0
    # ref = 
    temp = ''
    counter_tot = 0
    for s in range(len(source_topic)):
        a,b,c = list(chunk(source_topic[s],3))
        grams = gram_maker(reference[s], n_breaker)
        for elem in grams:
            if ' '.join(list(elem)) in a:
                one += 1
            elif ' '.join(list(elem)) in b:
                two += 1
            elif ' '.join(list(elem)) in c:
                three += 1
        counter_tot += len(grams)
    print(one/counter_tot,two/counter_tot,three/counter_tot)
    return 0

# Invert the segment positions
new_jumble = []
for s in range(len(source_topic)):
    a,b,c = list(chunk(source_topic[s],3))
    new_jumble.append(c+' '+b+' '+a)

# Write the jumbled source
with open(file_name,'w') as f:
    for elem in new_jumble:
        f.write(f'{elem}\n')
f.close()