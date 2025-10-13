import re
import nltk
import os
from config import parse_args
from inference import InferenceEngine
from inference import save_results
from method.prompt_processor import PromptProcessor
from utils import load_gzip, load_pkl, set_seed, log_hyperpara


nltk.download('punkt_tab')

def keep_top_n_evidence(retrieved_results, n):
    for qn_entity in retrieved_results:
        if len(qn_entity["top_k"])<n:
            continue
        qn_entity["top_k"] = qn_entity["top_k"][:n]
    return retrieved_results

def extract_domain(url):
    if url:
        domain_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        match = re.search(domain_pattern, url)
        if match:
            domain = match.group(1)
            return domain
    return url  


def Augment_w_Media(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = 'all'):
    available_media = 0
    unavailable_media = 0
    curated_media = 0
    for qn_entity in retrieved_results:
        for sentence_entity in qn_entity["top_k"]:
            domain = extract_domain(sentence_entity['original_link'])
            if media_data == 'mbfc':
                if domain in mbfc_credibility_data:
                    sentence_entity['details'] = " ".join(mbfc_credibility_data[domain]['details'])
                    available_media += 1
                else:
                    sentence_entity['details'] = "Not available."
                    unavailable_media += 1
            else: # using mbfc media data and AI generated media data
                if domain in all_credibility_data:
                    sentence_entity['details'] = all_credibility_data[domain]['details']
                    available_media += 1
                    if domain not in mbfc_credibility_data:
                        sentence_entity['details'] += "\n***The above descriptions are AI-generated without human verification and may not be fully reliable.***"
                        curated_media += 1
                else:
                    sentence_entity['details'] = "Not available."
                    unavailable_media += 1

    print("no of available media: ", available_media)
    print("no of AI generated media: ", curated_media)
    print("no of unavailable media: ", unavailable_media)

    return retrieved_results

def main():

    args = parse_args()

    set_seed(args.seed)

    results_folder = f'./results/results_{args.media_data}_media'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    qns_entities = load_gzip(args.source)
    all_credibility_data = load_pkl("./data/dataset/all_media_data.pkl")
    mbfc_credibility_data = load_pkl("./data/dataset/mbfc_media_data.pkl")
    retrieved_results = load_pkl(f'./results/top{args.n}_retrieved_{args.type}.pkl')

    model_name =  args.model.split('/')[-1]
    if args.with_MediaBG.lower() == "true":
        args.with_MediaBG = True
        print("MediaBG in use")
    else:
        args.with_MediaBG = False
        print("MediaBG not in use")

    log_hyperpara(args)

    file_name = f"Top_{args.k}_{args.type}_{args.method}_MediaBD_{args.with_MediaBG}_model_{model_name}"
    print(file_name)

    inference_result_file = os.path.join(results_folder, f"{file_name}_inference.txt")

    if args.method in ["DirectAnswer", "Explain", "CoT"]:

        retrieved_results = keep_top_n_evidence(retrieved_results, args.k)
        retrieved_results = Augment_w_Media(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = args.media_data)
                
        prompt_generator = PromptProcessor(
            prompt_type = args.method,
            with_MediaBG = args.with_MediaBG)

        prompts = prompt_generator.generate_prompts(retrieved_results)

        engine = InferenceEngine(args.model, args.seed, args.gpu)

        predictions = engine.infer(
            prompts = prompts, 
            method = args.method, 
            output_file = inference_result_file)
        

    elif args.method == "AgentBased":

        from method.agent_based import analysis_preprocess
        from method.agent_based import process_questions
        
        retrieved_results = keep_top_n_evidence(retrieved_results, args.k)
        retrieved_results = Augment_w_Media(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = args.media_data)

        qns, sentences, media_backgrounds = analysis_preprocess(retrieved_results)

        predictions = process_questions(qns, sentences, media_backgrounds, output_file = inference_result_file)


    elif args.method in ["RerankHard", "RerankSoft", "Filter"]:

        if args.method in ["RerankHard", "RerankSoft"]:
            from method.rerank import soft_rerank
            from method.rerank import hard_rerank

            results_file = os.path.join(results_folder, f'top{args.n}_rerank_scored_{args.type}.pkl')
            rerank_scored_results = load_pkl(results_file)

            if args.method == 'RerankSoft':
                processed_results = soft_rerank(rerank_scored_results, beta = 0.8)
            elif args.method == 'RerankHard':
                processed_results = hard_rerank(rerank_scored_results, beta = 0.8, gamma = 0.3)

        elif args.method == "Filter":
            from method.filter import filter_results

            processed_results = filter_results(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = args.media_data)
            
        processed_results = keep_top_n_evidence(processed_results, args.k)

        prompt_generator = PromptProcessor(
            prompt_type = 'DirectAnswer',
            with_MediaBG = False
            )

        prompts = prompt_generator.generate_prompts(processed_results)
    
        engine = InferenceEngine(args.model, args.seed, args.gpu)

        predictions = engine.infer(
            prompts = prompts, 
            method = "DirectAnswer", 
            output_file = inference_result_file)


    elif args.method == 'MajorityVoting':
        from method.majority_voting import analysis_preprocess
        from method.majority_voting import analyze_sentences
        from method.majority_voting import majority_voting

        retrieved_results = Augment_w_Media(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = args.media_data)

        qns, sentences, media_backgrounds = analysis_preprocess(retrieved_results)
        
        analysis_result_file = os.path.join(results_folder, f"Top_{args.k}_{args.type}_MajorityVoting_analysis.txt")
        analysis_results = analyze_sentences(qns, 
                                             sentences, 
                                             with_MediaBG = args.with_MediaBG, media_backgrounds = media_backgrounds,  output_file = analysis_result_file)

        predictions = majority_voting(analysis_results)

        
    final_result_file = os.path.join(results_folder, f"{file_name}.json")
    save_results(qns_entities, predictions, final_result_file)
    
if __name__ == "__main__":
    main()

