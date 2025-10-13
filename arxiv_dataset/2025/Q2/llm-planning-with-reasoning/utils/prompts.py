from utils.constants import TEMPLATE, INST_CODELLAMA_GEN,TEMPLATE_CORR, INST_CODELLAMA_ITER_CORR, INST_CUSTOM_GEN, INST_GEN_SIMPLE_DDL_MD_CHAT, TEMPLATE_SIMPLE_DDL_MD_CHAT, INST_GEN_REASONING, TEMPLATE_REASONING, TEMP_REASONING_EVAL_RES
from utils.constants import TEMP_REASONING_EVAL, TEMP_REASONING_EVAL_NOSCHEMA, TEMP_REASONING_EVAL_SCHEMA
from utils.normalize_sql import normalize_sql
import re


#%%  prompt generation

def prepare_eval_prompt(example):
    return TEMP_REASONING_EVAL.format(example['schema'], example['question'], example['sql'] )

def prepare_eval_promptV2(example,useSchema=False):
    if useSchema:
        return TEMP_REASONING_EVAL_SCHEMA.format(example['schema'], example['question'], example['sql'] )
    else:
        return TEMP_REASONING_EVAL_NOSCHEMA.format(example['question'], example['sql'] )

def prepare_eval_res_prompt(example):
    return TEMP_REASONING_EVAL_RES.format(example['schema'], example['question'], example['sql'], example['result'] )

def prepare_prompt(example, retriever_gen, prompt_method=0, headstr="SELECT"):
    # prepare prompt
    if retriever_gen is None:
        model_inp = TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    else:
        demos = retriever_gen.retrieve(example["question"])
        model_inp = "\n\n".join([TEMPLATE.format(ex["db_id"], ex["schema"], ex["question"]) + ex["sql"] for ex in demos])
        model_inp = model_inp + "\n\n" + TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    prompt = model_inp
    # add instruction
    if prompt_method == 0:
        prompt = INST_CODELLAMA_GEN.format(prompt) + " " + headstr
    elif prompt_method == 2:
        prompt = INST_CUSTOM_GEN.format(prompt) + " " + headstr
    return prompt, model_inp

# Impliments SIMPLE_DDL_MD_CHAT prompt format
def prepare_promptV2(example, retriever_gen=None, prompt_method=0, headstr=""):
    # prepare prompt
    model_inp = TEMPLATE_SIMPLE_DDL_MD_CHAT.format(example["schema"], example["question"])
    prompt = model_inp
    # add instruction
    prompt = INST_GEN_SIMPLE_DDL_MD_CHAT.format(prompt)
    return prompt, model_inp

# Impliments SIMPLE_DDL_MD_CHAT prompt format for reasoning
def prepare_promptVR(example, retriever_gen=None, prompt_method=0, headstr=""):
    # prepare prompt
    model_inp = TEMPLATE_REASONING.format(example["schema"], example["question"])
    prompt = model_inp
    # add instruction
    prompt = INST_GEN_REASONING.format(prompt)
    return prompt, model_inp

# Prompt the generator for 0-shot correction
def prepare_prompt_correction(example, answer_sql, prompt_method=0, headstr="SELECT"):
    prompt = TEMPLATE_CORR.format(example["db_id"], example["schema"], example["question"], answer_sql)
    if prompt_method == 0:
        prompt = INST_CODELLAMA_ITER_CORR.format(prompt) + " " + headstr
    return prompt

# prompt for tree
def prepare_prompt_step(model_inp, partial_sql, s="", prompt_method=0):
    if s != "":
        s = " " + s
    if prompt_method == 0:
        mc_prompt = INST_CODELLAMA_GEN.format(model_inp) + " " + partial_sql + s
    else:
        mc_prompt = model_inp + partial_sql
    return mc_prompt

#%% Extraction

# experimental extraction. Works so far.
def extract_completionsV2(responses, prompt, headstr):
    if headstr:
        header = headstr + " "
    else:
        header = ""
    # extract completions: collect text after prompt, add back any leading query (headstr) and before the next section (\n\n) if exists
    sql_completions = list(set([normalize_sql((header + r.split(prompt)[-1]).split("\n\n")[0]) \
                                for r in responses if (header + r.split(prompt)[-1]).split("\n\n")[0] != ""]))
    return sql_completions

# experimental extraction. Works so far.
def extract_completionsV3(responses, prompt, headstr):
    # if any leading part of query given in prompt
    if headstr:
        header = headstr + " "
    else:
        header = ""
    
    # End of SQL query.
    sqlend = ";"
    
    # Divider. Used to find the tail of the answer
    dividr = "\n\n"

    # if the divider is the end of query, we need to add it back.
    if dividr == sqlend:
        footer = sqlend
    else:
        footer = ""
    
    # Cleaning: List of characters to replace in the SQL query 
    rmlist = {
        "\n": "",
        "  ": " "
    }
    # extract completions: collect text after prompt, add back any leading query (headstr) and before End of SQL query
    sql_completions = list(set([normalize_sql((header + r.split(prompt)[-1]).split(dividr)[0] + footer) \
                                for r in responses if (header + r.split(prompt)[-1]).split(dividr)[0] + footer != ""]))
    # Clean the queries
    for i in range(len(sql_completions)):
        for key in rmlist.keys():
            sql_completions[i] = sql_completions[i].replace(key, rmlist[key])
        
        if sqlend not in sql_completions[i]:
           sql_completions[i] += sqlend 

    return sql_completions

# Look for answer in a particular format
def extract_completionsV4(responses):
    # End of SQL query.
    sqlend = ";"
    
    sql_start = "```sql\n"
    sql_end = "\n```"
    sql_format = f"{sql_start}(.*?){sql_end}"
    
    # Cleaning: List of characters to replace in the SQL query 
    rmlist = {
        "\n": "",
        "  ": " "
    }
    # extract completions: collect text after prompt, add back any leading query (headstr) and before End of SQL query
    sql_completions = list(set([normalize_sql(re.search(sql_format, r, re.DOTALL).group(1)) \
                                for r in responses if re.search(sql_format, r, re.DOTALL) ]))
    # Clean the queries
    for i in range(len(sql_completions)):
        for key in rmlist.keys():
            sql_completions[i] = sql_completions[i].replace(key, rmlist[key])
        
        if sqlend not in sql_completions[i]:
           sql_completions[i] += sqlend 

    # avoid empty answer
    if len(sql_completions)==0:
        sql_completions = [";"]
    return sql_completions

def extract_completions(responses,prompt_method=0):
    # extract completions: collect text after [-- SQL:] or [/INST] and before the next section (\n\n) if exists
    if prompt_method == 0:
        sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
    elif prompt_method == 1:
        sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))
    else:
        raise ValueError("Invalid method")
    return sql_completions