import logging
import random
import re
import tqdm
import json
logger = logging.getLogger(__name__)


def convert_string_to_unique_number(string: str) -> int:
  """
  Hack to convert SNLI ID into a unique integer ID, for tensorizing.
  """
  id_map = {'e': '0', 'c': '1', 'n': '2'}

  # SNLI-specific hacks.
  if string.startswith('vg_len'):
    code = '555'
  elif string.startswith('vg_verb'):
    code = '444'
  else:
    code = '000'

  try:
    number = int(code + re.sub(r"\D", "", string) + id_map.get(string[-1], '3'))
  except:
    number = random.randint(10000, 99999)
    logger.info(f"Cannot find ID for {string}, using random number {number}.")
  return number


def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  tsv_dict = {}

  i = -1
  with open(file_path, 'r') as tsv_file:
    for line in tqdm.tqdm([line for line in tsv_file]):
      i += 1
      if i == 0:
        header = line.strip()
        field_names = line.strip().split("\t")
        continue

      fields = line.strip().split("\t")
      label = fields[label_index]
      if len(fields) > len(field_names):
        # SNLI / MNLI fields sometimes contain multiple annotator labels.
        # Ignore all except the gold label.
        reformatted_fields = fields[:len(field_names)-1] + [label]
        assert len(reformatted_fields) == len(field_names)
        reformatted_line = "\t".join(reformatted_fields)
      else:
        reformatted_line = line.strip()

      if label == "-" or label == "":
        logger.info(f"Skippping line: {line}")
        continue

      if guid_index is None:
        guid = i
      else:
        guid = fields[guid_index] # PairID.
      if guid in tsv_dict:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
      tsv_dict[guid] = reformatted_line.strip()

  logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
    return tsv_numeric, header
  return tsv_dict, header

def read_glue_json(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  instances = []
  with open(file_path, 'r') as file:
    json_content = file.read()

  json_objects = json_content.split('\n')
  json_data = [json.loads(obj) for obj in json_objects if obj.strip()]

  # Iterate through each JSON entry
  for entry in json_data:
    # Extract relevant information
    question = entry.get("question", "")
    explanation = entry.get("exp", "")
    options = [entry.get("opa", ""), entry.get("opb", ""), entry.get("opc", ""), entry.get("opd", "")]
    subject_name = entry.get("subject_name", "")
    topic_name = entry.get("topic_name", "")
    question_id = entry.get("id", "")
    choice_type = entry.get("choice_type", "")

    # Create an instance dictionary
    instance = {
      "question": question,
      "explanation": explanation,
      "options": options,
      "subject_name": subject_name,
      "topic_name": topic_name,
      "question_id": question_id,
      "choice_type": choice_type
    }
    header = ", ".join(instance.keys())
    # Append the instance to the list
    instances.append(str(entry))

  return instances, header