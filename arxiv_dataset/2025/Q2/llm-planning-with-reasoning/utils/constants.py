TEMPLATE = "-- Database {}:\n{}\n-- Question: {}\n -- SQL:\n"
TEMPLATE_CORR = "-- Database {}:\n{}\n-- Question: {}\n -- Buggy SQL:\n{}\n -- Fixed SQL:\n"
TEMPLATE_SIMPLE_DDL_MD_CHAT = "{}\n\n### {}" # tables and question
TEMPLATE_REASONING = "{}\n\n### Question: {}" # tables and question
#TEMPLATE_REASONING = "{}\n### Question\n{}" # tables and question

TEMPLATE_EVAL = "-- Utterance: {}\n-- SQL:\n{}"
TEMPLATE_EVAL_RES = "-- Utterance: {}\n-- SQL:\n{}\n-- Result:\n{}"

INST_CODELLAMA_GEN = "[INST] Given database schema and a question in natural language, generate the corresponding SQL query.\n\n{} [/INST]"
INST_GEN_SIMPLE_DDL_MD_CHAT = "### Answer the question by sqlite SQL query only and with no explanation\n\n### Sqlite SQL tables, with their properties:\n{}\n### SQL:"
#INST_GEN_REASONING = "### Assume all tables given below exist in the SQL database and everything is properly set up. Answer the question by generating sqlite SQL query. Think logically step by step within <think></think> and be concise. Know the table names, column names, and relationships. Understand the question and schema, select only needed columns, use proper filtering. Avoid redundant joins and subqueries when possible. Your final answer must be a single valid SQL query that starts with SELECT and must not contain math operations (+,-,/,*). After providing final answer, do not describe or explain your answer again. \n\n### Sqlite SQL tables, with their properties:\n{}\n\n<think>\nOkay,"
#INST_GEN_REASONING = "### Answer the question by generating sqlite SQL query. Use simple logical steps and be concise. Final answer must be a single valid SQL query that starts with SELECT. No further explanation after the final answer. \n\n### Sqlite SQL tables, with their properties:\n{}\n\n<think>\nOkay,"
#INST_GEN_REASONING = "### Answer the question by generating sqlite SQL query. Final answer must be a single valid SQL query that starts with SELECT.\n\n### Sqlite SQL tables, with their properties:\n{}\n\n<think>\nOkay,"
#INST_GEN_REASONING = "### Create a single valid sqlite SQL query that answers the question. Final answer must start with SELECT.\n\n### Sqlite SQL tables, with their properties:\n{}\n\n<think>\nOkay,"
#INST_GEN_REASONING = "Generate a SQLite SQL query for the following question using the database schema. Use step-by-step reasoning to identify tables, columns, joins, and conditions.\n### Schema\n{}\n### Answer\nProvide the SQL query in a code block: ```sql [Final SQL query here] ```\n\n<think>\nOkay,"

# better
#INST_GEN_REASONING = "### Answer the question by generating sqlite SQL query.\n### Sqlite SQL tables, with their properties:\n{}\n### <think> Use step by step reasoning using the table names, column names, and relationships.</think>\n### <answer>Final answer must be a single valid SQL query that starts with SELECT. Final answer must be within ```sql and ``` </answer>\n\n<think>\nOkay,"
#INST_GEN_REASONING = " Answer the question. Final answer must be a single valid SQL query that starts with SELECT.\n### Schema:\n{}\n\n<think>\nOkay,"
INST_GEN_REASONING = "Answer the question by generating a valid sqlite SQL query. Final answer must start with SELECT.\n### Schema:\n{}\n\n<think>\nOkay,"


INST_CUSTOM_GEN = "Given database schema and a question in natural language, generate the corresponding SQL query.\n\n{} "

INST_CODELLAMA_ITER_CORR = "[INST] Given database schema and a question in natural language, correct the buggy SQL query and generate a fixed SQL query.\n\n{} [/INST]"

TEMP_REASONING_EVAL = "Is the SQL correct given the Schema and question? <answer> Provide Yes or No in bold. </answer>.\n### Schema:\n{}\n### Question:\n{}\n### SQL query:\n{}\n\n<think>\nOkay,"

TEMP_REASONING_EVAL_NOSCHEMA = """Determine whether the SQL query is correct or not for the given question. \
Provide final answer in JSON format with the key 'correct'.\
\n For this task do the following:\
\n- First, break down the query step by step\
\n- Then, determine if the query provides the answer to the given question.\
\nYou only need to determine in terms of correctness, if query is correct or not. Do not generate alternative query. 
\n### Question:\n{}\
\n### SQL query:\n{}\
\n\n<think>\nOkay,"""

TEMP_REASONING_EVAL_SCHEMA = """Determine whether the SQL query is correct or not for the given question. \
Provide final answer in JSON format with the key 'correct'.\
\n For this task do the following:\
\n- First, break down the query step by step. Use the Schema for this.\
\n- Then, determine if the query provides the answer to the given question.\
\nYou only need to determine in terms of correctness, if query is correct or not. Do not generate alternative query.\
\n\n### Schema:\n{}\
\n### Question:\n{}\
\n### SQL query:\n{}\
\n\n<think>\nOkay,"""

INST_CODELLAMA_EVAL = "[INST] Answer the following Yes/No question: Is the SQL correct given the utterance?\n\n{}\n-- Answer: [/INST]" 

INST_CODELLAMA_EVAL_DB = "[INST] Answer the following Yes/No question: Is the SQL correct given the database schema and the utterance?\n\n{}\n-- Answer: [/INST]"

INST_CODELLAMA_EVAL_RES = "[INST] Answer the following Yes/No question: Is the SQL correct given the utterance and its result?\n\n{}\n-- Answer: [/INST]"

TEMP_REASONING_EVAL_RES = "Is the SQL correct given the question and its result? <answer> Provide Yes or No in bold. </answer>.\n### Schema:\n{}\n### Question:\n{}\n### SQL query:\n{}\n### Result:\n{}\n\n<think>\nOkay,"

CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit', 'intersect', 'union', 'except', 'union all']
JOIN_KEYWORDS = ['join', 'on', 'as', 'right join', 'inner join', 'left join']
OTHER_KEYWORDS = ['distinct']
BIRD_KEYWORDS = ['if', 'else', 'datediff', 'over', 'instr', 'case', 'partition by', 'iif', 'float', 'real', 'when', 'int', 'using', 'timestampdiff', 'then', 'substr', 'cast', 'integer', 'strftime', 'end']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', 'not null', 'null']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(OTHER_KEYWORDS)
SQL_KEYWORDS.extend(BIRD_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)

TEMPLATE_GSM8K = "[INST] ## Given questions in the comment, use python programs to produce the correct answers with the 'answer' variable.\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n\n## {}\n [/INST] ## Python Program:\n"

TEMPLATE_GSM8K_EVAL = "[INST] ## Answer the following Yes/No question: Is the python program correct given the problem in the comment?\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n## Answer: Yes\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_per_tablet = 375\nn_tablets_per_day = 2\nn_tablets_per_6hrs = n_tablets_per_day / 6\nmg_per_6hrs = mg_per_tablet * n_tablets_per_6hrs\nanswer = mg_per_6hrs\n## Answer: No\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n## Answer: Yes\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\neggs_in_yard = 63\neggs_found_by_hannah = 2 * eggs_in_yard\neggs_found_by_helen = eggs_found_by_hannah / 2\nanswer = eggs_found_by_hannah\n## Answer: No\n\n## {}\n## Python Program:\n{}\n## Answer: [/INST]"

TEMPLATE_GSM8K_EVAL_RES = "[INST] ## Answer the following Yes/No question: Is the python program correct given its result and the problem in the comment?\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n## Result: 3000.0\n## Answer: Yes\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_per_tablet = 375\nn_tablets_per_day = 2\nn_tablets_per_6hrs = n_tablets_per_day / 6\nmg_per_6hrs = mg_per_tablet * n_tablets_per_6hrs\nanswer = mg_per_6hrs\n## Result: 125.0\n## Answer: No\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n## Result: 42\n## Answer: Yes\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\neggs_in_yard = 63\neggs_found_by_hannah = 2 * eggs_in_yard\neggs_found_by_helen = eggs_found_by_hannah / 2\nanswer = eggs_found_by_hannah\n## Result: 126\n## Answer: No\n\n## {}\n## Python Program:\n{}\n## Result: {}\n## Answer: [/INST]"

GSM8K_ITER_CORR = "[INST] ## Given the question in the comment, correct the buggy python program and generate a fixed python program to produce the correct answer with the 'answer' variable.\n\n## {}\n## Buggy Python Program:\n{}\n [/INST] ## Fixed Python Program:\n"