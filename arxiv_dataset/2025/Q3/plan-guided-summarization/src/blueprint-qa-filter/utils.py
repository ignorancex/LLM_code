
from datasets import load_dataset
import nltk
# from tqdm import tqdm
import re

def get_squality_summaries(split):
    squality_ds = load_dataset("pszemraj/SQuALITY-v1.3", split=split)
    questions = squality_ds["questions"]
    summaries_sents = []
    for i in range(4):
        for q in questions:
            sents = nltk.sent_tokenize(q[0]["responses"][i]["response_text"])
            summaries_sents.append(sents)
    return summaries_sents

def get_summaries(dataset_name, split):
    if dataset_name == "squality":
        return get_squality_summaries(split)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

# def get_summaries(dataset):
#     summaries = []
#     for example in dataset:
#         summary_text = example['summary']
#         summary_sents = [s.strip() for s in re.split(r'\[\d+\]', summary_text) if s.strip()]

#         num_sentences = int(re.search(r'\[(\d+)\]$', summary_text).group(1))
#         assert num_sentences == len(summary_sents), \
#             f"""Doc idx: {example['idx']}. Unexpected number of sentences:
#             {num_sentences} vs {len(summary_sents)} for summary sents: {summary_text}"""

#         summaries.append(summary_sents)
#     return summaries


# def extract_plan_point(plan_point):
#     match1 = re.search("^(\d+\.)(.*)(\[.*\])$", plan_point)
#     if match1:
#         return match1.group(2).strip()
    
#     match2 = re.search("^(\d+\.)(.*)(\[.*)$", plan_point)
#     if match2:
#         return match2.group(2).strip()
#     else:
#         raise ValueError(f"Couldn't match {plan_point}")

# def extract_plan_points(plan_points):
#     extracted_plan_points = []
#     for plan_point in plan_points.split("\n"):
#         extracted_plan_points.append(extract_plan_point(plan_point))
#     return extracted_plan_points

# def extract_point_reference(plan_point):
#     match1 = re.search("^(\d+\.)(.*)(\[.*\])$", plan_point)
#     if match1:
#         ref_match = match1.group(3).strip()[1:-1]
#         refs = [int(ref) for ref in ref_match.split(",")]
#         return refs
#     else:
#         raise ValueError(f"Couldn't match {plan_point}")

# def extract_point_references(plan_points):
#     extracted_point_references = []
#     for plan_point in plan_points.split("\n"):
#         extracted_point_references.append(extract_point_reference(plan_point))
#     return extracted_point_references
    

# def format_plan_point_qa_pair(idx, plan_point, questions, answers):
#     qa_formatted = " ".join([f"Q: {q} A: {a}" for q, a in zip(questions, answers)])
#     return (str(idx) + ". " + plan_point + " " + qa_formatted).strip()


# def format_coarse_plan_with_qa(plan_points, questions_all, answers_all):
#     assert len(plan_points) == len(questions_all) == len(answers_all), \
#         f"Lengths of plan points, questions, and answers do not match: {len(plan_points)} vs {len(questions_all)} vs {len(answers_all)}"    
#     formatted_plan_points = []
#     for i, (plan_point, questions, answers) in enumerate(zip(plan_points, questions_all, answers_all)):
#         formatted_plan_points.append(format_plan_point_qa_pair(i+1, plan_point, questions, answers))
#     return "\n".join(formatted_plan_points)


# def postprocess_summscreen_plans(coarse_plans):
#     problem_ids = [273,442,1087,1096,1100,1148,1149,1164,1179,1189,1198,1233,1235,1263,1279,1280,1326,1372,1432]
#     for plan_dict in coarse_plans:
#         if plan_dict['idx'] in problem_ids:
#             plan_with_citation = plan_dict['plan_with_citation']

#             # Step 1: Strip out hallucinated summary
#             clean_plan_match = re.match(r'(.*)Plan:(.*)', plan_with_citation, re.DOTALL)

#             if not clean_plan_match:
#                 raise ValueError(f"Could not extract plan from problematic ID: {plan_dict['idx']}, {plan_with_citation}")
            
#             clean_plan = clean_plan_match.group(2).strip()
            
#             # Step 2: Remove plan points not corresponding to summary sentences
#             summary_sent_len_match = re.match(r'(.*)\[(.*)\]$', plan_dict['summary'], re.DOTALL)
#             if not summary_sent_len_match:
#                 raise ValueError(f"Could not extract number of summary sentences from problematic ID: {plan_dict['idx']}, {plan_dict['summary']}")
#             summary_sent_len = int(summary_sent_len_match.group(2))
#             plan_points = clean_plan.split("\n")

#             cleaned_plan_points = []
#             for plan_point_with_ref in plan_points:
#                 try:
#                     point_refs = extract_point_reference(plan_point_with_ref)
#                 except ValueError:
#                     # Some badly formed plan points have no ref, so skip them.
#                     continue
#                 if max(point_refs) <= summary_sent_len:
#                     plan_point = extract_plan_point(plan_point_with_ref)
#                     cleaned_plan_points.append((plan_point, point_refs))
            
#             if not cleaned_plan_points:
#                 for plan_point_with_ref in plan_points:
#                     point_refs = extract_point_reference(plan_point_with_ref)
#                     if min(point_refs) <= summary_sent_len:
#                         plan_point = extract_plan_point(plan_point_with_ref)
#                         point_refs = [pr for pr in point_refs if pr <= summary_sent_len]
#                         cleaned_plan_points.append((plan_point, point_refs))
#             assert cleaned_plan_points, f"Doc Idx {plan_dict['idx']}: No clean plan points: {plan_points}"

#             # Step 3: create plan without citation
#             plan_dict['plan_with_citation'] = "\n".join([f"{i+1}. {plan_point} {refs}" for i, (plan_point, refs) in enumerate(cleaned_plan_points)])
#             plan_dict['plan_without_citation'] = "\n".join([f"{i+1}. {plan_point}" for i, (plan_point, _) in enumerate(cleaned_plan_points)])

#     return coarse_plans
