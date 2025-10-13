import os, sys
import torch

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def lexsort(orig_rows, tokenizer):
    input_ids = tokenizer(orig_rows, return_tensors='pt', padding=True)['input_ids']
    rows, cols = input_ids.shape
    indices = torch.arange(rows)
    for col in reversed(range(cols)):
        _, sorted_indices = torch.sort(input_ids[indices, col], stable=True)
        indices = indices[sorted_indices]
    reordered_rows = [orig_rows[i] for i in indices.tolist()]
    return reordered_rows

def parse_output(outputs):
        # ex: {"relevant": true, "best_answer": 1694}
        # to do, returns curr row was relevant and best answer
        #try:
        #    idx = pred.index('}')
    if "Answer" in outputs:
        top1_outputs = outputs.strip().replace('\n', '').split('}')[0] + "}"
        try:
            top1_outputs = eval(top1_outputs)['Answer']
            if type(top1_outputs) == list:
                top1_outputs = ', '.join(top1_outputs)
            if type(top1_outputs) == int or type(top1_outputs) == float:
                top1_outputs = str(top1_outputs)
            elif type(top1_outputs) != str:
                print(f"Type of output is {type(top1_outputs)}")
                print(f"As: {top1_outputs}")
                top1_outputs = str(top1_outputs)
        except:
            pass
    else:
        top1_outputs = outputs.strip().split('\n')[0]
    return top1_outputs

def parse_output_deprecated(pred):
    pred = pred.replace('null', 'None')

    try:
        out = eval(pred)
        answer = out['Answer']
    except:
        pred = pred.replace(',','')
        try:
            out = eval(pred)
        except:
            return "None"
    try:
        return str(out['Answer'])
    except:
        return str(out)

def linearize_table_rowwise(header, row):
    out = ''
    for c, r in zip(header, row):
        out += f"{c}: {r} | "
    if len(out) != 0:
        out = out[:-3] # remove last one's space and |
    else:
        out = 'None'
    return out


def group2chunks(l, n=5):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def remove_newline_and_convert_to_camelcase(headers):
    new_headers = []
    for value in headers:
        value = value.strip().replace('  ', ' ').replace('  ', ' ').replace('  ',' ')
        value = value.replace(' ', '_')
        value = value.replace('\n', '_')
        value = value.replace('__', '_').replace('__','_')
        new_headers.append(value)
    return new_headers

def get_table(texts, marker='question', table_marker=None):
    # break down into list of table info
    if table_marker is None:
        table_marker = 'table'
    prefix, the_rest = texts.split(f"{table_marker}:\n\n")
    prefix = prefix + f'{table_marker}:\n\n'
    the_rest, suffix = the_rest.split(f'\n\n\n{marker}:\n\n')
    suffix = f'\n\n{marker}:\n\n' + suffix
    table_info = the_rest.split(' row 1 : ')
    if len(table_info) == 3: # for feverous
        textual_info = table_info[2].split('\n')[-1]
        table_info[1] = table_info[1].split('col :')[0].strip()
    else:
        textual_info = ''
    columns = table_info[0].replace('col : ','').split(' | ')
    rows = []
    row_idx = 2
    if len(table_info) > 1:
        rows_raw = table_info[1]
        while True:
            try:
                index = rows_raw.index(f" row {row_idx} : ")
                row = rows_raw[:index]
                rows.append(row.split(' | '))
                rows_raw = rows_raw[index:].replace(f' row {row_idx} : ','')
                row_idx += 1
            except ValueError:
                # exception: if last row, last val consists of Null, we need to add space
                if rows_raw[-1] == '|':
                    rows_raw = rows_raw + ' '
                last_row = rows_raw.split(' | ')
                while len(last_row) < len(columns):
                    last_row.append(' ')
                rows.append(last_row)
                break

        if len(columns) != list(set([len(x) for x in rows]))[0]:
            print(f"length of column {len(columns)} and rows {[len(x) for x in rows]} doesn't match!!!")
            #import pdb; pdb.set_trace()
    # columns, rows conversion done
    # format first try: colname1: val1, colname2: val2\n...
    # columns: list of header, rows: list of list
    # convert '' to Null
    null_replaced_rows = []
    for rowline in rows:
        temp = []
        for x in rowline:
            if x == '':
                temp.append('null')
            else:
                temp.append(x)
        null_replaced_rows.append(temp)
    columns = remove_newline_and_convert_to_camelcase(columns)
    return {'header': columns, 'rows': null_replaced_rows, 'text': textual_info}



