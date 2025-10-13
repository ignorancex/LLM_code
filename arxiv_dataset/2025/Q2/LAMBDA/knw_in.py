import torch
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
# from knw import KNW_INJECTION, knowledge_injection
from prompt_engineering.prompts import PMT_KNW_IN_CORE, PMT_KNW_IN_FULL
# from config import rag_mode
from knowledge_integration.ncm import Nearest_Correlation_Matrix
from knowledge_integration.nn_network import nn_networks
from knowledge_integration.pami import pattern_mining
from kernel import execute


KNW_INJECTION = {}

def knowledge_register():
    ncm = Nearest_Correlation_Matrix()
    ncm_key = ncm.name+ncm.description
    KNW_INJECTION[ncm_key] = ncm
    nnn = nn_networks()
    nnn_key = nnn.name+nnn.description
    KNW_INJECTION[nnn_key] = nnn
    pami = pattern_mining()
    pami_key = pami.name+pami.description
    KNW_INJECTION[pami_key] = pami

embeding_model = SentenceTransformer('all-MiniLM-L6-v2')

def search_knowledge(user_input, knowledge_embeddings, knowledge_keys):
    input_embedding = embeding_model.encode(user_input, convert_to_tensor=True)
    similarities_list = util.pytorch_cos_sim(input_embedding, knowledge_embeddings)
    if torch.max(similarities_list) > 0.5:
        best_match_idx = np.argmax(similarities_list.cpu())
        best_match_key = knowledge_keys[best_match_idx]
    else:
        best_match_key = False
    return (best_match_key, KNW_INJECTION[best_match_key]) if best_match_key else (False, None)


def format_code_snaps(knw, kernel):
    desc = knw.description
    core_code = knw.get_core_function()
    if knw.mode == 'full':
        print("Knowledge_integration in full mode")
        return PMT_KNW_IN_FULL.format(desc=desc, code=knw.get_all_code())
    elif knw.mode == 'core':
        code_backend = knw.get_runnable_function()
        print("Knowledge_integration: core mode, runnable result: ", execute(code_backend,kernel))
        retrieval_knw = PMT_KNW_IN_CORE.format(desc=desc, core=core_code, code_backend=code_backend)
        return retrieval_knw
    else:
        return "Nothing retrieved, caused by the Invalid mode: {knw.mode}, please choose from ['full', 'core']."
        raise ValueError(f"Invalid mode: {knw.mode}, please choose from ['full', 'core'].")


def retrieval_knowledge(instruction, kernel): # return code_snaps and mode: 'full' or runnable code in 'core'. Nothing retrieval, return None
    knowledge_register()
    knowledge_keys = list(KNW_INJECTION.keys())
    knowledge_embeddings = embeding_model.encode(knowledge_keys, convert_to_tensor=True)
    best_key, best_knw_object = search_knowledge(instruction, knowledge_embeddings, knowledge_keys)
    if best_key:
        return format_code_snaps(best_knw_object, kernel)
    else:
        return None

