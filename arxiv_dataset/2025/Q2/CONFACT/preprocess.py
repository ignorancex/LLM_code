import re
from tqdm import tqdm
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
import pickle as pkl
from config import parse_args
from utils import load_gzip
import os

class Splitter:
    def __init__(self, method, max_length = 256):
        self.method = method
        self.max_length = max_length

    @staticmethod
    def split_into_sentences(text):
        abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'e.g', 'i.e']
        for abbr in abbreviations:
            text = re.sub(r'\b' + abbr + r'\.', abbr + '<ABBR>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [sentence.replace('<ABBR>', '.') for sentence in sentences]
        return sentences
    
    @staticmethod
    def split_into_chunks(text, max_length=256):
        sentences = Splitter.split_into_sentences(text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            # Count the total words if this sentence is added to the current chunk
            word_count = sum(len(chunk.split()) for chunk in current_chunk) + len(sentence.split())
            
            if word_count > max_length:
                # Start a new chunk if the word count exceeds max length
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
            else:
                # Otherwise, add the sentence to the current chunk
                current_chunk.append(sentence)

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process(self, source_path):
        data = load_gzip(source_path)
        
        all_evidence = {}
        
        for qn in tqdm(data, desc="Splitting evidence"):
            for evidence in qn['evidence_url']:
                evidence_id = evidence['evidence_id']
                evidence_url = evidence['original_link']
                content = evidence['content']
                
                # Split the content into chunks of max length while keeping sentences intact
                if self.method == 'sentences':
                    results = Splitter.split_into_sentences(content)
                elif self.method == 'chunks':
                    results = Splitter.split_into_chunks(content, self.max_length)
                
                all_evidence[evidence_id] = {
                    'url': evidence_url,
                    'sentences': results
                }
        return all_evidence
    
class BM25Retriever:
    def __init__(self, k=100):
        self.k = k

    def retrieve_for_qn(self, item, evidence_data):
        question_id = item['id']
        question = item['question']
        
        all_sentences = []
        sentence_corpus = []
        
        for evidence_id, content in evidence_data.items():
            if evidence_id.startswith(str(question_id) + '_'):
                url = content['url']
                sentences = content['sentences']
                all_sentences.extend(sentences)

                for sentence in sentences:
                    sentence_corpus.append({
                        'evidence_id': evidence_id,
                        'original_link': url,
                        'sentence': sentence
                    })

        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in all_sentences]
        bm25 = BM25Okapi(tokenized_sentences)

        qn_tokens = nltk.word_tokenize(question)
        scores = bm25.get_scores(qn_tokens)
        top_idx = np.argsort(scores)[::-1][:self.k]
        
        top_K = [
            {
                'sentence': sentence_corpus[i]['sentence'],
                'original_link': sentence_corpus[i]['original_link'],
                'score': scores[i]
            }
            for i in top_idx if scores[i] > 0
        ]

        return {
            'question_id': question_id,
            'question': question,
            'top_k': top_K,
        }

    def retrieve_for_batch(self, qns_entities, evidence_data, output_file):
        
        results = []
        for qn in tqdm(qns_entities, desc="Retrieving evidence for each question"):
            result = self.retrieve_for_qn(qn, evidence_data)
            results.append(result)

        # [{
        #     'question_id': question_id,
        #     'question': question,
        #     'top_k': top_K,
        # }, {}, ...
        # ]

        with open(output_file, 'wb') as f:
            pkl.dump(results, f)

        return results

def main():

    args = parse_args()
    results_folder = f'./results'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    nltk.download('punkt_tab')

    split_processor = Splitter(args.type, args.chunk_size)
    all_evidence = split_processor.process(args.source)

    qns_entities = load_gzip(args.source)

    retriever = BM25Retriever(args.n)
    store_path = f'./results/top{args.n}_retrieved_{args.type}.pkl'
    retriever.retrieve_for_batch(qns_entities, all_evidence, store_path)

if __name__ == "__main__":
    main()


