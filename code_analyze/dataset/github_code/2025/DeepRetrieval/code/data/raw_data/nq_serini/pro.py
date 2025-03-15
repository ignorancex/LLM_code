import json


with open("topics.nq.test.txt", "r") as f:
    lines = f.readlines()
    

jsonl_file = []
for line in lines:
    question = line.split("\t")[0]
    answers = line.split("\t")[1]
    # convert answers (e.g., ['Wilhelm Conrad Röntgen']) to list of strings
    answers = [answer.strip() for answer in answers.strip("[]").split(",")]
    jsonl_file.append({"question": question, "answers": answers})


with open("test.jsonl", "w") as f:
    for line in jsonl_file:
        f.write(json.dumps(line) + "\n")
