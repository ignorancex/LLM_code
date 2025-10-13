import jsonlines
import json
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="test_EN.jsonl", help="Input file to layout-gen-model")
parser.add_argument("--output_file", type=str, default="test_EN.jsonl", help="Output file from layout-gen-model")
parser.add_argument("--font_paths", nargs='+', default=["fonts/CourierPrime-Bold.ttf", "fonts/Roboto-Regular.ttf", "fonts/Neuton-Regular.ttf"], 
                    help="List of font paths to choose from. If you provide multiple paths, one will be randomly selected for each document rendering.")
parser.add_argument("--visualize_annt", type=bool, default=True,
                    help="Whether to render the visualized annotation image.")
args = parser.parse_args()


val_file = f"input_files/{args.input_file}" # input file to layout-gen-model
val_answer_file = f"output_files/{args.output_file}" # output file from layout-gen-model
annt_file = "contents//{}//annotations.json".format(".".join(os.path.basename(val_file).split('.')[:-1]))

print(f"Visualizing {val_answer_file}...")

output_dir = os.path.join("outputs", os.path.basename('.'.join((val_answer_file).split('.')[:-1])))
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

NEW_LINE_HEIGHT = 45
NEW_LINE_TITLE_HEIGHT = 60
H_OVERLAP_TOLERANCE = 200
W_TOLERENCE = 400 # right border expand tolerance
MARGIN_BORDER = 50
MARGIN_ENTITY = 20

# load annt
annt = json.load(open(annt_file, 'r', encoding='utf-8'))["forms"]
filename2annt = {x["filename"]: x["entities"] for x in annt}


def new_line(fill_id, cur_w, margin):
    """
    :param fill_id: entity to be modified
    :param cur_w: current width of document or left border of next entity (the limit of right border of this entity)
    :param margin: margin between document border or entities
    """
    
    text = id2text[fill_id]
    box = answer[fill_id]

    font_size = NEW_LINE_HEIGHT
    new_width = cur_w - margin - box[0]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))

    final_text = ""
    current_line = ""

    # split text into words
    words = text.split(' ')

    for word in words:
        test_line = current_line + ' ' + word if current_line != "" else word
        test_width = draw.textlength(test_line, font=ImageFont.truetype(font_path, font_size))
        
        if test_width > new_width and current_line != "":
            final_text += current_line + '\n'
            current_line = word
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

def new_line_header(fill_id, new_width):
    """
    This function is only for adjusting title to new line.

    :param fill_id: entity to be modified
    :param new_width: new width of this entity
    :return w, h: new width and height of this entity
    """
    
    text = id2text[fill_id]
    box = answer[fill_id]

    font_size = NEW_LINE_TITLE_HEIGHT
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))

    final_text = ""
    current_line = ""

    # split text into words
    words = text.split(' ')

    for word in words:
        test_line = current_line + ' ' + word if current_line != "" else word
        test_width = draw.textlength(test_line, font=ImageFont.truetype(font_path, font_size))
        
        if test_width > new_width and current_line != "":
            final_text += current_line + '\n'
            current_line = word
        else:
            current_line = test_line

    # add the last line
    final_text += current_line

    _, _, w, h = draw.textbbox((0, 0), final_text, font=ImageFont.truetype(font_path, font_size))

    id2text[fill_id] = final_text

    # move entities below <FILL_0> down
    move_y = box[1] + h - box[3]
    for key in answer.keys():
        if key == fill_id:
            continue
        if answer[key][1] > box[1]:
            answer[key][1] += move_y
            answer[key][3] += move_y

    return w, h

def check_out_of_bound(answer, width, height):
    """
    Check if the entities are out of bound of the document.
    """
    max_w = width
    max_h = height

    w_tolerance = W_TOLERENCE
    
    for key, value in answer.items():
        x1, y1, x2, y2 = value

        if x2 > max_w:
            if x2 - width < w_tolerance:
                max_w = x2
            else:
                x1, y1, x2, y2 = new_line(key, max_w, MARGIN_BORDER)
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

def filter_invalid(answer, width, height):
    """
    Filter out entities that are out of bound of the document or having invalid box.

    """
    for key, value in answer.items():
        x1, y1, x2, y2 = value
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            print(f"Error in {obj['filename']}: got invalid box {value} in {key}.")
            black_list.add(obj["filename"])
            return False
    return True


answers = list(jsonlines.open(val_answer_file, 'r'))
black_list = set()

with open(val_file, 'r', encoding='utf-8') as f:
    # compute lines
    num_lines = sum(1 for _ in f)
    f.seek(0)

    reader = jsonlines.Reader(f)
    for i, obj in enumerate(tqdm(reader, total=num_lines)):

        # control font
        font_path = np.random.choice(args.font_paths)
        header_bold = np.random.choice([True, False], p=[0.4, 0.6])
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
        except:
            try:
                if answers[i]["output"].find(">:") != -1: # invalid key format of dict
                    raw_answer = answers[i]["output"].split("</s>")[0].strip().replace(">:", '>":')
                    answer = json.loads(raw_answer)
                    for key, value in answer.items():
                        answer[key] = [int(num) for num in value.split(",")]
                else:
                    raise
            
            except:
                try: # correct end format
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
                except:
                    print(f"Error in {obj['filename']}")
                    black_list.add(obj["filename"])
                    continue

        flag = False
        origin_title_w = None
        # adjust gt box (directly modify box to fit all text (not adding new line yet))
        for j, (fill_id, text) in enumerate(id2text.items()):
            if fill_id in answer.keys():
                box = answer[fill_id]
            else:
                print(f"Error in {obj['filename']}: {fill_id} not found in answer.")
                black_list.add(obj["filename"])
                flag = True
                break
            draw = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
            
            # invalid box
            if box[1] >= box[3] or box[0] >= box[2] or box[0] < 0 or box[1] < 0:
                print(f"Error in {obj['filename']}: got invalid box {box} in {fill_id}.")
                black_list.add(obj["filename"])
                flag = True
                break

            _, _, w, h = draw.textbbox((0, 0), text, font=ImageFont.truetype(font_path, box[3] - box[1]))
            if j == 0: # header: center
                origin_title_w = box[2] - box[0]
                box = [(box[0] + box[2]) // 2 - w // 2, box[1], (box[0] + box[2]) // 2 + w // 2, box[3]]

                if box[0] < MARGIN_BORDER or box[2] > width - MARGIN_BORDER: # adjust header to new line
                    box_w, box_h = new_line_header("<FILL_0>", origin_title_w)
                    box = [(width - box_w) // 2, box[1], (width + box_w) // 2, box[1] + box_h]
            else:
                box = [box[0], box[1], box[0] + w, box[3]]
            answer[fill_id] = box
        if flag: # error in layout generation, skip this sample
            continue

        ## adjust overlap
        # sort answer by y
        answer = {k: v for k, v in sorted(answer.items(), key=lambda item: item[1][1])}

        rows = group_entities_by_row(answer)

        # fix horizontal overlap
        for row in rows:
            row_len = len(row)
            for j in range(row_len - 1):
                box1 = row[j][1]
                box2 = row[j + 1][1]
                if box1[0] < box2[0] and box1[2] > box2[0]:
                    overlap_length = box1[2] - box2[0]

                    # if overlap too much, adjust itself to new line first
                    if overlap_length > H_OVERLAP_TOLERANCE:
                        new_box = new_line(row[j][0], box2[0], MARGIN_ENTITY)
                        if new_box[2] <= box2[0]: # no overlap
                            continue
                        else:
                            overlap_length = new_box[2] - box2[0]
                    
                    # move box after j (start from j+1) to the right
                    for k in range(j + 1, row_len):
                        row[k][1][0] += overlap_length + 10
                        row[k][1][2] += overlap_length + 10

                        # will not overlap with the next entity, don't need to move the next entity
                        if k < row_len - 1: # ensure k+1 is valid
                            if row[k][1][2] < row[k+1][1][0]:
                                break

        max_w, max_h = check_out_of_bound(answer, width, height)
        if max_h != height:
            max_h += MARGIN_BORDER
        if max_w != width:
            max_w += MARGIN_BORDER
        width, height = max_w, max_h

        # adjust header to center again
        x1, y1, x2, y2 = answer["<FILL_0>"]
        answer["<FILL_0>"] = [(width - x2 + x1) // 2, y1, (width + x2 -x1) // 2, y2]

        if not filter_invalid(answer, width, height): # invalid box, skip this sample
            continue


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

            if entity_dict["text"].strip() != id2text[f"<FILL_{id}>"].replace('\n', " ").strip():
                print(f"Error in {obj['filename']}: {entity_dict['text']} !=", id2text[f'<FILL_{id}>'].replace('\n', ' '))
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
                if text.find('\n') == -1:
                    draw.text((box[0], box[1]), text, font=ImageFont.truetype(font_path, box[3] - box[1]), fill=color, stroke_width=1)
                else:
                    start_y = box[1]
                    for line_text in text.split('\n'):
                        _, _, w, h = draw.textbbox((0, 0), line_text, font=ImageFont.truetype(font_path, NEW_LINE_TITLE_HEIGHT))
                        draw.text(((box[0] + box[2]) // 2 - w // 2, start_y), line_text, font=ImageFont.truetype(font_path, NEW_LINE_TITLE_HEIGHT), fill=color, stroke_width=1) # center
                        start_y += NEW_LINE_TITLE_HEIGHT
            else:
                # header bold
                if label == "header" and header_bold:
                    stroke_width = 1
                else:
                    stroke_width = 0

                # checkbox related
                if font_path == "fonts/CourierPrime-Bold.ttf":
                    space_for_checkbox = ' '
                elif font_path == "fonts/Roboto-Regular.ttf":
                    space_for_checkbox = '   '
                elif font_path == "fonts/Neuton-Regular.ttf":
                    space_for_checkbox = '     '

                if text.find('\n') != -1:
                    if text.endswith(checkbox_empty) or text.endswith(checkbox_checked):
                        draw.text((box[0], box[1]), text.replace(checkbox_empty, '  ').replace(checkbox_checked, '  '), font=ImageFont.truetype(font_path, NEW_LINE_HEIGHT), fill=color, stroke_width=stroke_width)
                    else:
                        draw.text((box[0], box[1]), text.replace(checkbox_empty, space_for_checkbox).replace(checkbox_checked, space_for_checkbox), font=ImageFont.truetype(font_path, NEW_LINE_HEIGHT), fill=color, stroke_width=stroke_width)
                else:
                    if text.endswith(checkbox_empty) or text.endswith(checkbox_checked):
                        draw.text((box[0], box[1]), text.replace(checkbox_empty, '  ').replace(checkbox_checked, '  '), font=ImageFont.truetype(font_path, box[3] - box[1]), fill=color, stroke_width=stroke_width)
                    else:
                        draw.text((box[0], box[1]), text.replace(checkbox_empty, space_for_checkbox).replace(checkbox_checked, space_for_checkbox), font=ImageFont.truetype(font_path, box[3] - box[1]), fill=color, stroke_width=stroke_width)

            # draw checkbox
            if text.startswith(checkbox_empty) or text.startswith(checkbox_checked):
                if text[0] == checkbox_checked:
                    checkbox_text = '\u2612' # ☒
                elif text[0] == checkbox_empty:
                    checkbox_text = checkbox_empty

                if font_path == "fonts/CourierPrime-Bold.ttf":
                    space_between_text = 5
                elif font_path == "fonts/Roboto-Regular.ttf":
                    space_between_text = 0
                elif font_path == "fonts/Neuton-Regular.ttf":
                    space_between_text = 0

                if text.find('\n') == -1:
                    draw.text((box[0] + space_between_text, box[1]), checkbox_text, font=ImageFont.truetype("fonts/DejaVuSans.ttf", box[3] - box[1]) , fill=color)
                else:
                    draw.text((box[0] + space_between_text, box[1]), checkbox_text, font=ImageFont.truetype("fonts/DejaVuSans.ttf", NEW_LINE_HEIGHT) , fill=color)
            
            elif text.endswith(checkbox_empty) or text.endswith(checkbox_checked):
                if text[-1] == checkbox_checked:
                    checkbox_text = '\u2612' # ☒
                elif text[-1] == checkbox_empty:
                    checkbox_text = checkbox_empty

                draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
                if font_path == "fonts/CourierPrime-Bold.ttf":
                    space_between_text = 15
                elif font_path == "fonts/Roboto-Regular.ttf":
                    space_between_text = 5
                elif font_path == "fonts/Neuton-Regular.ttf":
                    space_between_text = 5
                if text.find('\n') == -1:
                    prepend_text_length = draw_temp.textlength(text[:-2], font=ImageFont.truetype(font_path, box[3] - box[1])) + space_between_text
                    draw.text((box[0] + prepend_text_length, box[1]), checkbox_text, font=ImageFont.truetype("fonts/DejaVuSans.ttf", box[3] - box[1]) , fill=color)
                else:
                    lines = text.split('\n')
                    last_line_text = lines[-1]
                    prepend_text_length = draw_temp.textlength(last_line_text[:-2], font=ImageFont.truetype(font_path, NEW_LINE_HEIGHT)) + space_between_text
                    draw.text((box[0] + prepend_text_length, box[1] + (len(lines)-1)*NEW_LINE_HEIGHT), checkbox_text, font=ImageFont.truetype("fonts/DejaVuSans.ttf", NEW_LINE_HEIGHT) , fill=color)

            # store linking
            if args.visualize_annt:
                for link in entity["linking"]:
                    box1 = answer[f"<FILL_{link[0]}>"]
                    box2 = answer[f"<FILL_{link[1]}>"]
                    relation_lines.append([((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2), (box2[0], box2[1])])
        
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
                draw.line(line, fill=(69, 139, 0), width=4)

            with_annt.save(os.path.join(output_dir, "visualize", os.path.basename(obj['filename']).split(".")[0] + ".jpg"))

# write black list to txt
black_list = sorted(list(black_list))
with open(os.path.join(output_dir, "black_list.txt"), 'w') as f:
    for item in black_list:
        f.write("%s\n" % item)