import sys
from modify_movieqa import read_qa_json

qas = read_qa_json('data/data/qa.json', split='val')
labels = [qa.correct_index for qa in qas]
f = open("evaluation_file.txt", "x")
counter = 0
for l in labels:
    f.write("test:" + str(counter) + " " + int(l))
    counter = counter + 1
f.close()
