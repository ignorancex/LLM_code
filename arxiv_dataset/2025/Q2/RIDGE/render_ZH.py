import jsonlines
import json
import os
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
import re
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="test_TC.jsonl", help="Input file to layout-gen-model")
parser.add_argument("--output_file", type=str, default="test_TC.jsonl", help="Output file from layout-gen-model")
parser.add_argument("--font_paths", nargs='+', default=["fonts/GuanKiapTsingKhai.ttf"], 
                    help="List of font paths to choose from. If you provide multiple paths, one will be randomly selected for each document rendering.")
parser.add_argument("--visualize_annt", type=bool, default=True,
                    help="Whether to render the visualized annotation image.")
args = parser.parse_args()


val_file = f"input_files/{args.input_file}" # input file to layout-gen-model
val_answer_file = f"output_files/{args.output_file}" # output file from layout-gen-model
annt_file = "contents//{}//annotations.json".format(".".join(os.path.basename(val_file).split('.')[:-1]))

print(f"Visualizing {val_answer_file}...")

output_dir = "outputs/" + ".".join(os.path.basename(val_answer_file).split('.')[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, "images")):
    os.makedirs(os.path.join(output_dir, "images"))
if not os.path.exists(os.path.join(output_dir, "annotations")):
    os.makedirs(os.path.join(output_dir, "annotations"))
if args.visualize_annt and not os.path.exists(os.path.join(output_dir, "visualize")):
    os.makedirs(os.path.join(output_dir, "visualize"))


color_map = {
    "header": (224, 102, 255),
    "question": (240, 128, 128),
    "answer": (79, 148, 205),
    "other": (255, 165, 0),
}

# define ☐ and ☑ 's unicode
checkbox_empty = '\u2610'
checkbox_checked = '\u2611'

MARGIN_BORDER = 50
W_TOLERENCE = 400
NEW_LINE_HEIGHT = 40

# load annt
annt = json.load(open(annt_file, 'r', encoding='utf-8'))["forms"]
filename2annt = {x["filename"]: x["entities"] for x in annt}


def adjust_answer(x1, y1, x2, y2): # fix negative height
    if y2 < y1:
        if (y2 % 1000) > (y1 % 1000):
            y2 = (y1 // 1000) * 1000 + (y2 % 1000)
        else:
            y2 = (y1 // 1000 + 1) * 1000 + (y2 % 1000)

    return x1, y1, x2, y2

def check_answer(x1, y1, x2, y2):
    error_mes = None
    if x1 > x2:
        error_mes = f"negative width: {x1} > {x2}"
    if y1 > y2:
        error_mes = f"negative height: {y1} > {y2}"

    return error_mes

def new_line(fill_id, cur_w):
    text = id2text[fill_id]
    box = answer[fill_id]

    margin = MARGIN_BORDER
    font_size = NEW_LINE_HEIGHT
    new_width = cur_w - margin - box[0]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))

    final_text = ""
    current_line = ""

    for char in text:
        test_line = current_line + char
        test_width = draw.textlength(test_line, font=ImageFont.truetype(font_path, font_size))
        
        if test_width > new_width:
            final_text += current_line + '\n'
            current_line = char
        else:
            current_line = test_line

    # add the last line
    final_text += current_line

    _, _, w, h = draw.textbbox((0, 0), final_text, font=ImageFont.truetype(font_path, font_size))

    id2text[fill_id] = final_text
    answer[fill_id] = [box[0], box[1], box[0] + w, box[1] + h]

    # move entities below <FILL_ID> down
    move_y = answer[fill_id][3] - box[3]
    col_id = -1
    for i, row in enumerate(rows):
        if fill_id not in [x[0] for x in row]:
            continue
        else:
            col_id = i
            break

    for i in range(col_id + 1, len(rows)):
        for entity in rows[i]:
            entity_id = entity[0]
            answer[entity_id][1] += move_y
            answer[entity_id][3] += move_y

    return answer[fill_id]

def check_out_of_bound(answer, width, height):
    max_w = width
    max_h = height

    w_tolerance = W_TOLERENCE
    
    for key, value in answer.items():
        x1, y1, x2, y2 = value

        if x2 > max_w:
            if x2 - width < w_tolerance:
                max_w = x2
            else:
                need_line_height_fix.add(int(key.replace("<FILL_", "").replace(">", "")))
                x1, y1, x2, y2 = new_line(key, max_w)
                if x2 > max_w:
                    max_w = x2

        if y2 > max_h:
            max_h = y2
    
    return max_w, max_h

def group_entities_by_row(answer, row_tolerance=20):
    rows = []
    current_row = []
    previous_y = 0

    for key, value in answer.items():
        x1, y1, x2, y2 = value
        
        if not current_row:
            current_row.append((key, value))
            previous_y = y1
        else:
            if abs(y1 - previous_y) <= row_tolerance:
                current_row.append((key, value))
            else:
                current_row = sorted(current_row, key=lambda x: x[1][0]) # sort by x
                rows.append(current_row)
                current_row = [(key, value)]
                previous_y = y1
                
    if current_row:
        current_row = sorted(current_row, key=lambda x: x[1][0]) # sort by x
        rows.append(current_row)
        
    return rows


answers = list(jsonlines.open(val_answer_file, 'r'))
black_list = set()

with open(val_file, 'r', encoding='utf-8') as f:
    # compute lines
    num_lines = sum(1 for _ in f)
    f.seek(0)

    reader = jsonlines.Reader(f)
    for i, obj in enumerate(tqdm(reader, total=num_lines)):

        # font control
        font_path = np.random.choice(args.font_paths)
        header_bold = np.random.choice([True, False], p=[0.5, 0.5])
        # header_bold = False

        input_dict = json.loads(obj['input'])
        width = input_dict["width"]
        height = input_dict["height"]

        drop_ids = [] # drop entity whose text is "_____"
        id2text = {}
        for x in input_dict["entities"]:
            id2text[x["box"][0]] = x["text"]
            if x["text"].strip().replace("_", "") == "":
                drop_ids.append(int(x["box"][0].split('_')[-1][:-1]))
        last_fill_id = int(list(id2text.keys())[-1].split('_')[-1][:-1])

        # filter out the string not in json format
        xs = answers[i]["output"].split("</s>")[0].strip().split(",")
        to_be_removed = []
        for x in xs:
            if re.fullmatch(r' \".*\"', x):
                to_be_removed.append(x)
        for string in to_be_removed:
            xs.remove(string)
        if len(to_be_removed) > 0:
            answers[i]["output"] = ",".join(xs)

        # load answer as dict
        try:
            answer = json.loads(answers[i]["output"].split("</s>")[0].strip())
            for key, value in answer.items():
                answer[key] = [int(num) for num in value.split(",")]
                if answer[key][3] < answer[key][1]: # y2 < y1
                    print(f"Adjust height for {obj['filename']}:\nText: {id2text[key]}\nBox: {answer[key]}")
                    answer[key] = [pos for pos in adjust_answer(*answer[key])]
                err_mes = check_answer(*answer[key])
                if err_mes:
                    print(f"Error in {obj['filename']}: {err_mes}")
                    black_list.add(obj["filename"])
                    continue
                
        except:
            try:
                raw_answer = answers[i]["output"]
                target = f'\"<FILL_{last_fill_id}>\"'
                pos = raw_answer.find(target)
                if pos == -1:
                    raise
                end_pos = raw_answer[pos + len(target):].find('\"') + pos + len(target)
                end_pos = raw_answer[end_pos + 1:].find('\"') + end_pos + 1
                if end_pos == -1:
                    raise
                answer = json.loads((raw_answer[:end_pos + 1] + '}').strip())
                for key, value in answer.items():
                    answer[key] = [int(num) for num in value.split(",")]
                    if answer[key][3] < answer[key][1]: # y2 < y1
                        print(f"Adjust height for {obj['filename']}:\nText: {id2text[key]}\nBox: {answer[key]}")
                        answer[key] = [pos for pos in adjust_answer(*answer[key])]
                    err_mes = check_answer(*answer[key])
                    if err_mes:
                        print(f"Error in {obj['filename']}: {err_mes}")
                        black_list.add(obj["filename"])
                        continue
            except:
                print(f"Error in {obj['filename']}")
                black_list.add(obj["filename"])
                continue

        flag = False

        # adjust gt box
        for j, (fill_id, text) in enumerate(id2text.items()):
            if fill_id in answer.keys():
                box = answer[fill_id]
            else:
                print(f"Error in {obj['filename']}: {fill_id} not found in answer.")
                black_list.add(obj["filename"])
                flag = True
                break
            draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
            _, _, w, h = draw.textbbox((0, 0), text, font=ImageFont.truetype(font_path, box[3] - box[1]))
            if j == 0: # header: center
                draw.text(((box[0] + box[2]) // 2 - w // 2, box[1]), text, font=ImageFont.truetype(font_path, box[3] - box[1]), fill="black", stroke_width=1)
                box = [(box[0] + box[2]) // 2 - w // 2, box[1], (box[0] + box[2]) // 2 + w // 2, box[3]]
            else:
                draw.text((box[0], box[1]), text, font=ImageFont.truetype(font_path, box[3] - box[1]), fill="black")
                box = [box[0], box[1], box[0] + w, box[3]]
            answer[fill_id] = box
        if flag: # error in layout generation, skip this sample
            continue

        ## adjust overlap
        # sort answer by y
        answer = {k: v for k, v in sorted(answer.items(), key=lambda item: item[1][1])}
        
        rows = group_entities_by_row(answer)
        # horizontal overlap
        for row in rows:
            row_len = len(row)
            for j in range(row_len - 1):
                box1 = row[j][1]
                box2 = row[j + 1][1]
                if box1[0] < box2[0] and box1[2] > box2[0]:
                    overlap_length = box1[2] - box2[0]
                    for k in range(j + 1, row_len):
                        row[k][1][0] += overlap_length + 10
                        row[k][1][2] += overlap_length + 10

        need_line_height_fix = set()
        max_w, max_h = check_out_of_bound(answer, width, height)
        if max_h != height:
            max_h += MARGIN_BORDER
        if max_w != width:
            max_w += MARGIN_BORDER
        width, height = max_w, max_h
        # adjust header to center again
        x1, y1, x2, y2 = answer["<FILL_0>"]
        answer["<FILL_0>"] = [(width - x2 + x1) // 2, y1, (width + x2 -x1) // 2, y2]


        ## write annotation
        document_dict = {
            "width": width,
            "height": height,
            "form": []
        }

        entities = []
        drop_entities = [] # (text, box) # we need this since we still need to draw the dropped entities in the image
        for entity_dict in filename2annt[obj['filename'].split(".")[0]]:
            id = entity_dict["id"]
            if id in drop_ids:
                drop_entities.append((id2text[f"<FILL_{id}>"], answer[f"<FILL_{id}>"]))
                continue

            if entity_dict["text"] != id2text[f"<FILL_{id}>"].replace('\n', ""):
                print(f"Error in {obj['filename']}: {entity_dict['text']} != {id2text[f'<FILL_{id}>']}")
                black_list.add(obj["filename"])
                flag = True
                break

            # drop links in drop_ids
            linking = entity_dict["linking"]
            linking = [x for x in linking if x[0] not in drop_ids and x[1] not in drop_ids]

            new_entity_dict = {
                "id": id,
                "text": id2text[f"<FILL_{id}>"],
                "box": answer[f"<FILL_{id}>"],
                "label": entity_dict["label"],
                "linking": linking
            }
            entities.append(new_entity_dict)
        if flag:
            continue
        # sort entities by id
        entities = sorted(entities, key=lambda x: x["id"])
        document_dict["form"] = entities
        json.dump(document_dict, open(os.path.join(output_dir, "annotations", f'{obj["filename"].split(".")[0]}.json'), 'w', encoding="utf8"), ensure_ascii=False)
        

        ## draw image
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(background)
        if args.visualize_annt:
            mask = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            mask_draw = ImageDraw.Draw(mask)

        relation_lines = []
        alpha = 0.3
        
        for entity in entities:
            id = entity["id"]
            text = entity["text"]
            box = entity["box"]
            label = entity["label"]

            # draw rectangle
            if args.visualize_annt:
                color = color_map[label] + (int(255 * alpha),)
                mask_draw.rectangle([box[0], box[1], box[2], box[3]], fill=color)

            # draw text
            color = (0, 0, 0, 255)
            if id == 0: # header: bold
                if font_path.find("GuanKiapTsingKhai") != -1:
                    y_shift = 0
                else:
                    y_shift = (box[3] - box[1]) // 4
                draw.text((box[0], box[1] - y_shift), text, font=ImageFont.truetype(font_path, box[3] - box[1]), fill=color, stroke_width=1)
            else:
                font_path_temp = font_path
                if label == "header" and header_bold:
                    if font_path.find("Regular") != -1:
                        font_path_temp = font_path.replace("Regular", "SemiBold")
                        stroke_width = 0
                    else: # "GuanKiapTsingKhai" only has regular, but stroke_width=1 is weird
                        stroke_width = 0
                else:
                    stroke_width = 0

                if text.find('\n') != -1 or id in need_line_height_fix:
                    if font_path_temp.find("GuanKiapTsingKhai") != -1:
                        y_shift = 0
                    else:
                        y_shift = NEW_LINE_HEIGHT // 4
                    draw.text((box[0], box[1] - y_shift), text.replace(checkbox_empty, '　').replace(checkbox_checked, '　'), font=ImageFont.truetype(font_path_temp, NEW_LINE_HEIGHT), fill=color, stroke_width=stroke_width)
                else:
                    if font_path_temp.find("GuanKiapTsingKhai") != -1:
                        y_shift = 0
                    else:
                        y_shift = (box[3] - box[1]) // 4
                    draw.text((box[0], box[1] - y_shift), text.replace(checkbox_empty, '　').replace(checkbox_checked, '　'), font=ImageFont.truetype(font_path_temp, box[3] - box[1]), fill=color, stroke_width=stroke_width)

            # draw checkbox
            if text.startswith(checkbox_empty) or text.startswith(checkbox_checked):
                if text.find('\n') == -1:
                    draw.text((box[0] + 5, box[1]), text[0], font=ImageFont.truetype("fonts/DejaVuSans.ttf", box[3] - box[1]) , fill=color)
                else:
                    draw.text((box[0] + 5, box[1]), text[0], font=ImageFont.truetype("fonts/DejaVuSans.ttf", NEW_LINE_HEIGHT) , fill=color)
            elif text.endswith(checkbox_empty) or text.endswith(checkbox_checked):
                draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
                if text.find('\n') == -1:
                    prepend_text_length = draw_temp.textlength(text[:-1], font=ImageFont.truetype(font_path, box[3] - box[1]))
                    draw.text((box[0] + prepend_text_length, box[1]), text[-1], font=ImageFont.truetype("fonts/DejaVuSans.ttf", box[3] - box[1]) , fill=color)
                else:
                    lines = text.split('\n')
                    last_line_text = lines[-1]
                    prepend_text_length = draw_temp.textlength(last_line_text[:-1], font=ImageFont.truetype(font_path, NEW_LINE_HEIGHT))
                    draw.text((box[0] + prepend_text_length, box[1] + (len(lines)-1)*NEW_LINE_HEIGHT), text[-1], font=ImageFont.truetype("fonts/DejaVuSans.ttf", NEW_LINE_HEIGHT) , fill=color)

            # store linking
            if args.visualize_annt:
                for link in entity["linking"]:
                    box1 = answer[f"<FILL_{link[0]}>"]
                    box2 = answer[f"<FILL_{link[1]}>"]
                    relation_lines.append([((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2), (box2[0], box2[1]), (box2[0], box2[1])])
        
        # draw dropped entities
        for text, box in drop_entities:
            draw.text((box[0], box[1]), text, font=ImageFont.truetype(font_path, box[3] - box[1]), fill=(0, 0, 0, 255))

        if args.visualize_annt:
            with_annt = Image.alpha_composite(background, mask)

        background = background.convert("RGB")
        background.save(os.path.join(output_dir, "images", os.path.basename(obj['filename']).split(".")[0] + ".jpg"))

        # blend background and mask
        if args.visualize_annt:
            with_annt = with_annt.convert("RGB")
            draw = ImageDraw.Draw(with_annt)
            for line in relation_lines:
                draw.line(line, fill=(69, 139, 0), width=6)

            with_annt.save(os.path.join(output_dir, "visualize", os.path.basename(obj['filename']).split(".")[0] + ".jpg"))


# write black list to txt
black_list = sorted(list(black_list))
with open(os.path.join(output_dir, "black_list.txt"), 'w') as f:
    for item in black_list:
        f.write("%s\n" % item)