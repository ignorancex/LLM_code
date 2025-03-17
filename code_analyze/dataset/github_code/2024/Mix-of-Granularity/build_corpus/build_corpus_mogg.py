# Import libraries
import json
import os
import sys
from tqdm import tqdm
import torch

medrag_path = 'MoG/src'
sys.path.insert(0, medrag_path)

from config import config
from tqdm import tqdm
import json
os.environ['HF_ENDPOINT'] = config['hf_endpoint']
os.environ['HF_HUB_URL'] = config['hf_hub_url']

from concurrent.futures import ThreadPoolExecutor

from utils import Retriever

# Manually define some parameters that used to be in the argparse
qa_dataset_name = 'medqa'

# Read the paths defined in the config file
prediction_folder, cache_dir, benchmark_repo_dir, benchmark_dataset_json, medmcqa_path, bioasq_path, pubmedqa_path, medqa_path, mmlu_path, tensorboard_log_dir, router_checkpoint_path, retrieval_result_path \
    = \
config['prediction_folder'], config['cache_dir'], config['benchmark_repo_dir'], config['benchmark_dataset_json'], config['medmcqa_path'], config['bioasq_path'], config['pubmedqa_path'], config['medqa_path'], config['mmlu_path'], config['tensorboard_log_dir'], config['router_checkpoint_path'], config['retrieval_result_path']

# Overwrite some arguments if needed
db_dir, db_moe_dir, db_graph_dir = config['db_dir'], config['db_moe_dir'], config['db_graph_dir']

medmcqa_path = None if "medmcqa" not in qa_dataset_name.lower() else medmcqa_path
bioasq_path = None if "bioasq" not in qa_dataset_name.lower() else bioasq_path
pubmedqa_path = None if "pubmedqa" not in qa_dataset_name.lower() else pubmedqa_path
medqa_path = None if "medqa" not in qa_dataset_name.lower() else medqa_path
mmlu_path = None if "mmlu" not in qa_dataset_name.lower() else mmlu_path

# define the cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

half_path = os.path.join(db_graph_dir, 'wikipedia_half', 'chunk')
con_1_path = os.path.join(db_graph_dir, 'wikipedia_con_1', 'chunk')
con_2_path = os.path.join(db_graph_dir, 'wikipedia_con_2', 'chunk')
graph_1_path = os.path.join(db_graph_dir, 'wikipedia_graph_1', 'chunk')
graph_2_path = os.path.join(db_graph_dir, 'wikipedia_graph_2', 'chunk')

new_path_list = [half_path, con_1_path, con_2_path, graph_1_path, graph_2_path]
for path in new_path_list:
    if not os.path.exists(path):
        os.makedirs(path)

def merge_json_objects(queue):
    # bug fix: the id should be a string without prefix #
    id_list = [json_object['id'].split("#")[1] for json_object in queue]
    content_list = [json_object['content'] for json_object in queue]
    new_id = '|'.join(id_list)
    new_content = ' '.join(content_list)
    json_object_new = {'id': new_id, 'content': new_content, 'title': queue[0]['title'], 'contents': queue[0]['title']+"."+new_content}
    return json_object_new



# Read the source Textbook corpora
textbook_copora_dir = os.path.join(db_moe_dir, 'wikipedia_half', 'chunk')
# loop over all the files under textbook_copora_dir
textbook_copora_files = []
for file_name in os.listdir(textbook_copora_dir):
    if file_name.endswith('.jsonl'):
        textbook_copora_files.append(file_name)

print("textbook_copora_files")
print(textbook_copora_files)


# -------------------------------------------------------------
# Build wikipedia_con_1, wikipedia_con_2, wikipedia_half
# -------------------------------------------------------------        
for i1 in tqdm(range(len(textbook_copora_files)), desc='files'):
    file_name = textbook_copora_files[i1]
    file_json_obj_list = []
    with open(os.path.join(textbook_copora_dir, file_name), 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                file_json_obj_list.append(data)
            except:
                continue
    
    # the same as half
    half_file_path = os.path.join(half_path, file_name)
    with open(half_file_path, 'w+') as f:
        counter = 0
        for i in tqdm(range(len(file_json_obj_list)), desc= 'Writing half', leave=False):
            j_object = file_json_obj_list[i].copy()
            old_id = j_object['id']
            j_object['id'] = str(counter+1)+'#'+old_id
            f.write(json.dumps(j_object)+'\n')
            counter += 1
            
    # wikipedia_con_1
    # contains the context within distance 1
    con_1_file_path = os.path.join(con_1_path, file_name)
    with open(con_1_file_path, 'w+') as f:
        counter = 0
        for i in tqdm(range(1, len(file_json_obj_list)-1, 3), desc = 'Writing con_1', leave=False):
            j_object = file_json_obj_list[i].copy()
            j_object_prev = file_json_obj_list[i-1].copy()
            j_object_next = file_json_obj_list[i+1].copy()
            j_object_merge = merge_json_objects([j_object_prev, j_object, j_object_next])
            old_id = j_object_merge['id']
            j_object_merge['id'] = str(counter+1)+'#'+old_id
            f.write(json.dumps(j_object_merge)+'\n')
            counter += 1
            
    # wikipedia_con_2
    # contains the context within distance 2
    con_2_file_path = os.path.join(con_2_path, file_name)
    with open(con_2_file_path, 'w+') as f:
        counter = 0
        for i in tqdm(range(2, len(file_json_obj_list)-2, 5), desc = 'Writing con_2', leave=False):
            j_object = file_json_obj_list[i].copy()
            j_object_prev_2 = file_json_obj_list[i-2].copy()
            j_object_prev_1 = file_json_obj_list[i-1].copy()
            j_object_next_1 = file_json_obj_list[i+1].copy()
            j_object_next_2 = file_json_obj_list[i+2].copy()
            j_object_merge = merge_json_objects([j_object_prev_2, j_object_prev_1, j_object, j_object_next_1, j_object_next_2])
            old_id = j_object_merge['id']
            j_object_merge['id'] = str(counter+1)+'#'+old_id
            f.write(json.dumps(j_object_merge)+'\n')
            counter += 1
            


# -------------------------------------------------------------
# Build wikipedia_graph_1, wikipedia_graph_2
# -------------------------------------------------------------   
retriever_name = "bm25"
        
# PARAMETERS
CREATE_GRAPH_CORPORA = True
INDEX_GRAPH_CORPORA = True

# Read the source Textbook corpora
textbook_copora_dir = os.path.join(db_moe_dir, 'wikipedia_half', 'chunk')
# loop over all the files under textbook_copora_dir
textbook_copora_files = []
for file_name in os.listdir(textbook_copora_dir):
    if file_name.endswith('.jsonl'):
        textbook_copora_files.append(file_name)


if CREATE_GRAPH_CORPORA:
    # Initialize a retriever
    # define the parameters for the retriever
    
    db_dir = db_dir + "/corpus_mog"
    corpus = "wikipedia_half"
    retriever_cache_dir = os.path.join(cache_dir, 'retriever_cache')
    do_moe = True
    threshold = 0
    router_threshold = 50

    retriever = Retriever(retriever_name, corpus, db_dir, retriever_cache_dir = cache_dir, do_moe=do_moe, threshold=threshold, router_threshold=router_threshold)

    def single_build_graph_corpus(file_json_obj_list, j, graph_1_file_path, graph_2_file_path):
        snippet = file_json_obj_list[j]
        snippet_id = snippet['id']
        snippet_content = snippet['content']
        
        # retrieve the snippets that are within distance 1
        query = snippet_content
        retrieved_snippets = retriever.get_relevant_documents(query, k=3)[0] #[1] would be the scores
        
        if retrieved_snippets[0] != 'NO_TEXT_RETRIEVED':
            
            # filter the redudant snippet, ideally, the first one should be the query snippet
            retrieved_snippets = [snip for snip in retrieved_snippets if snip['id'] != snippet_id]
            
            
            # keep the top two snippets if there are more than 2 snippets
            if len(retrieved_snippets) > 2:
                retrieved_snippets = retrieved_snippets[:2]
            corpus_graph_1_list = retrieved_snippets.copy()   
                
            # further "extend" these top 2 candidates by doing one more retreival
            # this is to build corpus_graph_2
            corpus_graph_2_list = retrieved_snippets.copy()
            for k in range(len(retrieved_snippets)):
                snippet_2 = retrieved_snippets[k]
                snippet_id_2 = snippet_2['id']
                snippet_content_2 = snippet_2['content']
                
                # retrieve the snippets that are within distance 1
                query_2 = snippet_content_2
                retrieved_snippets_2 = retriever.get_relevant_documents(query_2, k=2)[0]
                
                if retrieved_snippets_2[0] != 'NO_TEXT_RETRIEVED':
                    # filter the redudant snippet, ideally, the first one should be the query snippet
                    retrieved_snippets_2 = [snip for snip in retrieved_snippets_2 if snip['id'] != snippet_id_2 and snip['id'] != snippet_id]
                    # keep the top 1 snippets if there are more than 1 snippets
                    if len(retrieved_snippets_2) > 1:
                        retrieved_snippets_2 = retrieved_snippets_2[:1]
                else:
                    retrieved_snippets_2 = []
                
                corpus_graph_2_list.extend(retrieved_snippets_2)        
        else:
            corpus_graph_1_list, corpus_graph_2_list = [], []
            
        # merge the retrieved snippets for corpus_graph_1
        corpus_graph_1_list.insert(0, snippet)
        merged_snippet = merge_json_objects(corpus_graph_1_list)

        # write the merged snippet to the file, this is corpus_graph_1
        with open(graph_1_file_path, 'a') as f:
            f.write(json.dumps(merged_snippet)+'\n')
            
        # merge the retrieved snippets for corpus_graph_2
        corpus_graph_2_list.insert(0, snippet)
        merged_snippet = merge_json_objects(corpus_graph_2_list)
            
        # write the merged snippet to the file, this is corpus_graph_2
        with open(graph_2_file_path, 'a') as f:
            f.write(json.dumps(merged_snippet)+'\n')
            
        return 0


    for i1 in tqdm(range(len(textbook_copora_files)), desc='files'):
        file_name = textbook_copora_files[i1]
        file_json_obj_list = []
        with open(os.path.join(textbook_copora_dir, file_name), 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    file_json_obj_list.append(data)
                except:
                    continue
            
        # wikipedia_graph_1 contains the context within distance 1  
        graph_1_file_path = os.path.join(graph_1_path, file_name)
        # wikipedia_graph_2 contains the context within distance 2  
        graph_2_file_path = os.path.join(graph_2_path, file_name)
        
        # at the beginning, create an empty file
        with open(graph_1_file_path, 'w+') as f:
            pass
        
        # Define the maximum number of threads
        max_threads = 16

        progress_bar = tqdm(total=len(file_json_obj_list))
        threads = []  # List to store the thread objects
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            
            for j in range(len(file_json_obj_list)):
                future = executor.submit(single_build_graph_corpus, file_json_obj_list, j, graph_1_file_path, graph_2_file_path)
                futures.append(future)
                
            # Wait for the futures to complete and update the progress bar
            for future in futures:
                future.result()
                progress_bar.update(1)
            
        # Close the progress bar
        progress_bar.close()
        
        
if INDEX_GRAPH_CORPORA:
    # Index the graph corpora
    # Add the line number (1-index) to the field 'id', separated with '#'
    
    for i1 in tqdm(range(len(textbook_copora_files)), desc='files'):
        file_name = textbook_copora_files[i1]
        
        # wikipedia_graph_1 contains the context within distance 1  
        graph_1_file_path = os.path.join(graph_1_path, file_name)
        # wikipedia_graph_2 contains the context within distance 2  
        graph_2_file_path = os.path.join(graph_2_path, file_name)
        
        graph_corpus_path_list = [graph_1_file_path, graph_2_file_path]
        
        for j in tqdm(range(len(graph_corpus_path_list)), leave=False, desc='graph_corpus'):
            graph_corpus_path = graph_corpus_path_list[j]
            file_json_obj_list = []
            with open(graph_corpus_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        file_json_obj_list.append(data)
                    except:
                        continue
                    
            for k in tqdm(range(len(file_json_obj_list)), leave=False, desc='json_obj'):
                file_json_obj_list[k]['id'] = str(k+1) + '#' + file_json_obj_list[k]['id']
            
            # write the updated json objects to the file
            with open(graph_corpus_path, 'w') as f:
                for line in file_json_obj_list:
                    f.write(json.dumps(line)+'\n')
                