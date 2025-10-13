import sys
sys.path.append("..")
from utils import extract_domain

def filter_results(retrieved_results, all_credibility_data, mbfc_credibility_data, media_data = 'all'):
    revised_retrieved_results = []

    for qn_entity in retrieved_results:
        revised_top_k = []

        for sentence_entity in qn_entity["top_k"]:
            domain = extract_domain(sentence_entity['original_link'])
            if media_data == 'all':
                if all_credibility_data[domain]['credibility']!= 'low':
                    revised_top_k.append(sentence_entity)
            else:
                if domain in mbfc_credibility_data:
                    if mbfc_credibility_data[domain]['credibility']!= 'low':
                        revised_top_k.append(sentence_entity)

                else:
                    revised_top_k.append(sentence_entity)

        qn_entity["top_k"] = revised_top_k
        revised_retrieved_results.append(qn_entity)


    return revised_retrieved_results
