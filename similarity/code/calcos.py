import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_code_similarities(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['submission_id', 'sim_AC_ANS', 'sim_AC_REF', 'sim_ANS_REF'])
        sims_ac_ans = []
        sims_ac_ref = []
        sims_ans_ref = []
        for item in data:
            sid = item.get('submission_id')
            ac_code = item.get('sourceCode', '')
            ans_code = item.get('generate_code_block', '')
            ref_code = item.get('generate_ref_code_block', '')
            vect = TfidfVectorizer().fit([ac_code, ans_code, ref_code])
            vecs = vect.transform([ac_code, ans_code, ref_code])
            sim_matrix = cosine_similarity(vecs)
            sim_ac_ans = sim_matrix[0, 1]
            sim_ac_ref = sim_matrix[0, 2]
            sim_ans_ref = sim_matrix[1, 2]
            writer.writerow([sid, f'{sim_ac_ans:.4f}', f'{sim_ac_ref:.4f}', f'{sim_ans_ref:.4f}'])
            sims_ac_ans.append(sim_ac_ans)
            sims_ac_ref.append(sim_ac_ref)
            sims_ans_ref.append(sim_ans_ref)
        avg_ac_ans = sum(sims_ac_ans) / len(sims_ac_ans) if sims_ac_ans else 0.0
        avg_ac_ref = sum(sims_ac_ref) / len(sims_ac_ref) if sims_ac_ref else 0.0
        avg_ans_ref = sum(sims_ans_ref) / len(sims_ans_ref) if sims_ans_ref else 0.0
        writer.writerow(['average', f'{avg_ac_ans:.4f}', f'{avg_ac_ref:.4f}', f'{avg_ans_ref:.4f}'])
if __name__ == '__main__':
    compute_code_similarities('deepseek_32b_cpp_extract.json', 'deepseek_32b_cpp_sim_cosine.csv')