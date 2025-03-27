from english_words import english_words_set
import random

random.seed(1)

# random english words
english_words = sorted(english_words_set)
english_words = list(english_words)
random.shuffle(english_words)

# save the list
file = open('src/dataforcorruptions/randomwords.txt','w')
for words in english_words:
	file.write(words+"\n")
file.close()


# frequent words in wikipedia https://github.com/IlyaSemenov/wikipedia-word-frequency/blob/master/results/enwiki-2022-08-29.txt
freq_file_raw = open('src/dataforcorruptions/frequentwords-raw.txt','r')
freq_file = open('src/dataforcorruptions/frequentwords.txt','w')

raw_lines = freq_file_raw.readlines()

for line in raw_lines:
	word = line.split(' ')[0]
	freq_file.write(word+"\n")

freq_file_raw.close()
freq_file.close()
