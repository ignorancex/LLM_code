from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import json

def load_retriever(model_name, corpus_path):
    if model_name == 'openAI':
        API_KEY = ""
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=API_KEY,
        )
    else:
        embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda:0", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
        )
    
    vectorstore = FAISS.load_local(corpus_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    return retriever


if __name__ == '__main__':
	
	model = 'openAI'
    
    queries = [
        'Chi è il professore di Intelligenza Artificiale 1 per il corso di Laurea Magistrale in Ingegneria Informatica?',
        'quali sono le scadenze di iscrizione ad un corso di laurea?',
        'come funziona la magistrale di chimica?',
        'ciao! sono un ragazzo appena uscito dal liceo che è interessato al settore legale, in particolare alle leggi sulle aziende. dove potrei iscrivermi?',
        'come posso prenotare un appuntamento in segreteria?',
        'Come si pagano le tasse?'
    ]

    path = 'unipa-gpt/' 	# path till main folder

	retriever = load_retriever(model_name = model, corpus_path = path+'corpora/embeddongs/corpus-clear/')
    qa_retrieved = []
    
    for query in queries:
        print('Query:', query)
        answer = retriever.invoke(query)
        print('Results:', answer)
        qa_retrieved.append((query, answer))

    fixed_qa_retrieved = []
    for i in range(len(qa_retrieved)):
        dict_list = [dict(x) for x in qa_retrieved[i][1]]
        fixed_qa_retrieved.append((qa_retrieved[i][0], dict_list))

    with open(path+'corpora/qa_retrieved_'+model+'.json', 'w') as f:
        json.dump(fixed_qa_retrieved, f, indent=4)
