import json
import jsonlines
import glob
import re
import os
import pandas as pd
import shutil
import argparse


class Node:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.label = ""
        self.linking = []

        self.parent = None
        self.children = []

def traverse(node, annotation):
    if len(node.children) > 0:
        if node.label != "header":
            node.label = "question" # "answer -> question"
        for child in node.children:
            node.linking.append([node.id, child.id])
            child.linking.append([node.id, child.id])
            traverse(child, annotation)

    if node.parent is None and len(node.children) == 0 and node.label != "header":
        node.label = "other"


    annotation["entities"].append({
        "id": node.id,
        "text": node.text,
        "label": node.label,
        "linking": node.linking
    })


def main(args):
    src_name = args.content_dir
    content_dir = f'contents/{src_name}'

    filenames = sorted(glob.glob(f'{content_dir}/*.txt'))
    if not os.path.exists("input_files"):
        os.makedirs("input_files")
    writer = jsonlines.open(f"input_files/{src_name}.jsonl", mode='w')

    num_file_skip = 0
    for blacklist_file in glob.glob(f'{content_dir}/blacklist/*.txt'):
        filenames.remove(os.path.join(content_dir, os.path.basename(blacklist_file)))
        num_file_skip += 1
    new_blacklist = [] # (error message, filename)
    blacklist_dir = f"{content_dir}/blacklist"

    annotations = {"src": src_name, "forms": []}

    for filename in filenames:
        with open(filename, 'r', encoding="utf8") as f:
            data_output = {
                "src": src_name,
                "filename": os.path.basename(filename),
                "input": "",
            } 

            annotation = {
                "filename": os.path.basename(filename).split('.')[0],
                "entities": []
            }
            # build tree
            head_list = []
            pre_nodes = [None] * 10 # head, -, --, ---, ...
            pre_nodes_other = [None] * 10 # head, -, --, ---, ...
            cur_status = -1 # 0: head, 1: -, 2: --, 3: ---

            lines = f.readlines()

            # remove '\n' at the end of each line
            lines = [line.strip() for line in lines]

            # title
            header = lines[0]
            entities = [{
                "text": header,
                "box": ["<FILL_0>"]}]
            annotation["entities"].append({
                "id": 0,
                "text": header,
                "label": "header",
                "linking": []})
            entity_cnt = 1
            
            other_flag = True

            error_flag = False
            for i, line in enumerate(lines[1:]):
                cur_status = 0
                while(line.startswith("-")):
                    cur_status += 1
                    line = line[1:]
                line = line.strip()

                if line == "<content>" or line == "</content>" or line == "":
                    continue

                elif line.startswith("<h"):
                    entities.append({
                        "text": line.split(">")[-1].strip(), # not "> " since some will be "<h1>title"
                        "box": [f"<FILL_{entity_cnt}>"]})
                    
                    node = Node(entity_cnt, line.split(">")[-1].strip())
                    node.label = "header"
                    node.parent = None

                    head_list.append(node)
                    pre_nodes[0] = node
                    
                    entity_cnt += 1

                    other_flag = False

                elif line.startswith("</h"):
                    pre_nodes = [None] * 10
                    other_flag = True
                    pre_nodes_other = [None] * 10


                # key: value
                elif re.match(r".*: .*", line) and not (line.find(':') > line.find('(') and line.find(':') < line.find(')')):
                    entities.append({
                        "text": line.split(": ")[0] + ":",
                        "box": [f"<FILL_{entity_cnt}>"]})
                    
                    node = Node(entity_cnt, line.split(": ")[0] + ":")
                    node.label = "question"
                    if other_flag:
                        if cur_status == 1:
                            node.parent = None
                            head_list.append(node)
                        else:
                            node.parent = pre_nodes_other[cur_status - 1]
                            node.parent.children.append(node)
                    else:
                        node.parent = pre_nodes[cur_status - 1]
                        node.parent.children.append(node)

                    if other_flag:
                        pre_nodes_other[cur_status] = node
                    else:
                        pre_nodes[cur_status] = node
                    entity_cnt += 1
                    
                    # checkboxes are written in one line
                    num_check_box = line.split(": ")[1].count("☑") + line.split(": ")[1].count("☐")
                    if num_check_box > 1:
                        if line.split(": ")[1][0] == '☑' or line.split(": ")[1][0] == '☐':
                            check_boxes = re.findall(r"☑ .*?(?= ☐| ☑|$)|☐ .*?(?= ☐| ☑|$)", line.split(": ")[1])
                        else:
                            check_boxes = re.findall(r'\b\w+\s[☑☐]', line.split(": ")[1])
                            check_boxes = [check_box.strip() for check_box in check_boxes]
                        # print(check_boxes)
                        for check_box in check_boxes:
                            if re.match(r".*: .*", check_box):
                                entities.append({
                                    "text": check_box.split(": ")[0] + ":",
                                    "box": [f"<FILL_{entity_cnt}>"]})
                                
                                node_q = Node(entity_cnt, check_box.split(": ")[0] + ":")
                                node_q.label = "question"
                                node_q.parent = node
                                node.children.append(node_q)

                                entity_cnt += 1

                                entities.append({
                                    "text": check_box.split(": ")[1],
                                    "box": [f"<FILL_{entity_cnt}>"]})
                                
                                node_a = Node(entity_cnt, check_box.split(": ")[1])
                                node_a.label = "answer"
                                node_a.parent = node_q
                                node_q.children.append(node_a)

                                entity_cnt += 1

                            else:
                                entities.append({
                                    "text": check_box,
                                    "box": [f"<FILL_{entity_cnt}>"]})
                                
                                node_a = Node(entity_cnt, check_box)
                                if check_box[-1] == ':':
                                    node_a.label = "question"
                                else:
                                    node_a.label = "answer"
                                node_a.parent = node
                                node.children.append(node_a)

                                entity_cnt += 1
                    else:
                        entities.append({
                            "text": ': '.join(line.split(": ")[1:]), # in case there are multiple ':' in the value
                            "box": [f"<FILL_{entity_cnt}>"]})
                        
                        node_a = Node(entity_cnt, ': '.join(line.split(": ")[1:]))
                        node_a.label = "answer"
                        node_a.parent = node
                        node.children.append(node_a)

                        entity_cnt += 1
                elif re.match(r".*: .*", line) and line.count(':') > 1:
                    print(f"ERROR in {filename}: {line}")
                    new_blacklist.append((f"Parsing ':' error", f"{filename} {i+2}"))
                    num_file_skip += 1
                    shutil.copy(filename, f"{blacklist_dir}/{os.path.basename(filename)}")
                    error_flag = True
                    break
                    # exit()
                    

                # single key or single value
                else:
                    entities.append({
                        "text": line,
                        "box": [f"<FILL_{entity_cnt}>"]})
                    
                    node = Node(entity_cnt, line)
                    if line[-1] == ':':
                        node.label = "question"
                    else:
                        node.label = "answer" # may be changed afterwards if has children
                    if other_flag:
                        if cur_status == 1:
                            node.parent = None
                            head_list.append(node)
                        else:
                            node.parent = pre_nodes_other[cur_status - 1]
                            node.parent.children.append(node)
                    else:
                        node.parent = pre_nodes[cur_status - 1]
                        node.parent.children.append(node)

                    if other_flag:
                        pre_nodes_other[cur_status] = node
                    else:
                        pre_nodes[cur_status] = node
                    entity_cnt += 1
                    
            if error_flag:
                continue


            input_dict = {
                "width": 2480,
                "height": 3508,
                "entities": entities,
            }
            data_output["input"] = json.dumps(input_dict, ensure_ascii=False)

            writer.write(data_output)

            # write annotation (traverse tree)
            for head in head_list:
                traverse(head, annotation)

            # sort annotation["entities"] by id
            annotation["entities"] = sorted(annotation["entities"], key=lambda x: x["id"])
            annotations["forms"].append(annotation)

    writer.close()
    json.dump(annotations, open(f"{content_dir}/annotations.json", "w", encoding="utf8"), ensure_ascii=False)

    print("Number of files skipped:", num_file_skip)

    # add new blacklist to the original blacklist
    df = pd.read_csv(f"{content_dir}/blacklist/blacklist.csv")
    new_df = pd.DataFrame(new_blacklist, columns=["error_message", "filename (line)"])
    df = pd.concat([df, new_df], ignore_index=True) # concatenate new_df to df
    df.to_csv(f"{content_dir}/blacklist/blacklist.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default="example")
    args = parser.parse_args()

    main(args)