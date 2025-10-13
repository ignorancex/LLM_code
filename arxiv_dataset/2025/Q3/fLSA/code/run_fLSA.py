import json
import os
import random
import re
import sys
from nltk.tokenize import sent_tokenize

# Choose your own LLM
MODEL = ""

# Hyperparameters
N_tags = 100
em_steps = 30
n_e_batches = 500
e_batch_size = 1
batch_size = 10
N_train_samples = 1000

# Project path
project_path = "../"
# train_data_dir can be either a file path string or a list of file paths
train_data_dir = ""
# output log files
tag_filename = ""
assignment_filename = ""


# Example Data Loading Function
def loadData(data_dir, n_samples=N_train_samples, n_loaded_segments=0):
    # Input:
    #   data_dir: a single file path
    #   n_samples: maximum number of samples to be loaded from this file
    #   n_loaded_segments: number of segments loaded so far
    # Output:
    #   dataset: list of document strings (in this case problem and solution pair)
    #   n_segments: total number of segments loaded so far
    n_segments = n_loaded_segments
    dataset = []
    for file in os.listdir(data_dir):
        filename = os.path.join(data_dir, os.fsdecode(file))
        with open(filename, "r") as f:
            item = json.load(f)
            data_string = "<problem>\n" + item["problem"] + "\n</problem>\n"
            data_string += "<solution>\n"
            for sent in sent_tokenize(item["solution"]):
                n_segments += 1
                data_string += "Segment x" + str(n_segments) + ": " + sent + "\n"
            data_string += "</solution>"
            dataset.append(data_string)
            if len(dataset) >= n_samples:
                break
    return dataset, n_segments


def sampleResponse(prompt, model=MODEL, temperature=1, max_tokens=1000, top_p=0.5):
    # Input:
    #   prompt: text prompt for the LLM call
    #   model: name of the LLM
    #   temperature: LLM sampling temperature
    #   max_tokens: maximum number of output tokens
    #   top_p: LLM sampling top_p
    # Output:
    #   response_text: text response from the LLM call
    pass

# Initialize tag descriptions
tags = ['']
for i in range(N_tags):
    tags[0] += f"Tag {i+1}: This is Tag {i+1}.\n"

# Load the training data
if isinstance(train_data_dir, str):
    grouped_segments, N_segments = loadData(train_data_dir)
else:
    grouped_segments, N_segments = [], 0
    for dir in train_data_dir:
        n_samples = min(N_train_samples - len(grouped_segments), N_train_samples // len(train_data_dir))
        segs, n = loadData(dir, n_samples=n_samples, initial_n_segments=N_segments)
        grouped_segments += segs
        N_segments = n

N_batches = len(grouped_segments)

# Initialize from existing log files if any
starting_iter = 0
if os.path.isfile(tag_filename):
    with open(tag_filename, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith("Iteration"):
                starting_iter = int(line.split()[-1]) + 1
                tags.append("")
            elif line.startswith("Tag"):
                tags[-1] += line + "\n"
else:
    with open(tag_filename, "x") as file:
        file.write(tags[0])

    with open(assignment_filename, "x") as file:
        file.write("Assignments\n")

# EM algorithm
print("Starting from Iteration %d" % starting_iter)
for step in range(starting_iter, em_steps):
    seg_to_tag = {}
    # E-step
    if step == 0:
        for seg_id in range(1, N_segments + 1):
            tag_id = random.choice([f"Tag {i+1}" for i in range(N_tags)])
            seg_to_tag[f"Segment x{seg_id}"] = tag_id
    else:
        if step == em_steps - 1:
            n_e_batches = 1e6
        seg_ids = list(range(N_batches))
        random.shuffle(seg_ids)
        for i in range(0, min(n_e_batches, len(seg_ids)), e_batch_size):
            batch = ""
            for j in range(i, min(i + e_batch_size, len(seg_ids))):
                batch += grouped_segments[seg_ids[j]] + "\n"
            # Perform the E-step
            prompt = """Task: For each segment, find the tag best describing the segment.
            Example 1:
            Below are several math problems and their corresponding solution segments:
            <problem>
            Find the area in square feet of a square with a perimeter of 32ft.
            </problem>
            <solution>
            Segment x1: If the perimeter of the square is $32$ feet, then the length of each side is $\\frac{32}{4}=8$ feet.
            Segment x2: That makes the area of the square $8^2=\\boxed{64}$ square feet.
            </solution>

            The tags are:
            Tag 1: Compute the area of the sqaure.
            Tag 2: Compute the perimeter of the square.
            Tag 3: Compute the length of all sides.

            Repeat each solution segment and then assign a tag from the above tag list to the segment:
            Segment x1: If the perimeter of the square is $32$ feet, then the length of each side is $\\frac{32}{4}=8$ feet.
            Assignment: Tag 3: Compute the length of all sides.
            Segment x2: That makes the area of the square $8^2=\\boxed{64}$ square feet.
            Assignment: Tag 1: Compute the area of the sqaure.
            
            Example 2:
            """
            prompt += "Below are several math problems and their corresponding solution segments:\n"
            prompt += batch
            prompt += "The tags are:\n"
            prompt += tags[step]
            prompt += "Repeat each solution segment and then assign a tag from the above tag list to the segment:\n"
            estep = sampleResponse(prompt, max_tokens=1000)
            print(estep)

            # Extract tag assignments
            current_segment = ""
            for line in estep.split("\n"):
                line = line.strip()
                if line.startswith("Segment"):
                    current_segment = line.split(":")[0]
                elif line.startswith("Assignment:") and current_segment:
                    current_tag = line.split(":")[1].strip()
                    seg_to_tag[current_segment] = current_tag
                    current_segment = ""

    # M-step
    tag_descriptions = []
    for i in range(1, N_tags + 1):
        tag_id = "Tag " + str(i)
        # Collect all segments assigned to tag_id
        tagged_batch = []
        for segment in grouped_segments:
            seg_ids = re.findall("Segment x[0-9]+", segment)
            tag_ids = [seg_to_tag[seg_id] if seg_id in seg_to_tag else "NoTag" for seg_id in seg_ids]
            if tag_id in tag_ids:
                tagged_segment = segment
                for seg_id in seg_ids:
                    if seg_id in seg_to_tag:
                        tagged_segment = tagged_segment.replace(seg_id, seg_id + " - " + seg_to_tag[seg_id])
                tagged_batch.append(tagged_segment + "\n")
        if len(tagged_batch) == 0:
            tag_descriptions.append(tags[-1].split("\n")[i - 1])
            continue
        if step == 0:
            _batch_size = 1
        else:
            _batch_size = batch_size
        if len(tagged_batch) > _batch_size:
            tagged_batch = random.sample(tagged_batch, k=_batch_size)
        tagged_batch = "\n".join(tagged_batch)

        # Perform the M-step
        prompt = "The goal is to find a description for each tag, given the solution steps belonging to each tag."
        prompt += "Below are segments of math problem solutions, followed by their corresponding tag numbers:\n"
        prompt += tagged_batch
        prompt += """For each tag number, aggregate all the segments associated with that number,
            then associate to that tag a one-sentence summarization that describes the common purpose of all these segments.
        """
        prompt += "Write your answer as Tag ID: plot description.\n"
        mstep = sampleResponse(prompt)
        print(prompt)
        print(mstep)

        # Extract tag descriptions
        tag_description = tags[-1].split("\n")[i - 1]
        for line in mstep.split("\n"):
            line = line.strip()
            if line.startswith("Tag"):
                tag = list(re.findall("Tag [0-9]+", line))
                if tag and tag[0] == tag_id:
                    tag_description = line
                    break
        tag_descriptions.append(tag_description)
        
    tags.append("\n".join(tag_descriptions))
    print("Iteration " + str(step))
    print("-------------")
    print(tags[-1])
    print("\n")

    with open(tag_filename, "a") as file:
        file.write("Iteration " + str(step) + "\n")
        file.write(tags[-1])
        file.write("\n")
        file.close()
    with open(assignment_filename, "a") as file:
        file.write("Iteration " + str(step) + "\n")
        for seg_id, tag_id in seg_to_tag.items():
            file.write(seg_id + " - " + tag_id + "\n")
        file.write("\n")
        file.close()
