import json
import csv
import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarities(data, csv_path):
    """
    计算 TF-IDF + 余弦相似度
    :param data: JSON 数据列表
    :param csv_path: 输出 CSV 文件路径
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['submission_id', 'sim_AC_ANS', 'sim_AC_REF', 'sim_ANS_REF'])
        sims_ac_ans, sims_ac_ref, sims_ans_ref = [], [], []

        for item in data:
            sid = item.get('submission_id')
            ac, ans, ref = item.get('sourceCode', ''), item.get('generate_code_block', ''), item.get('generate_ref_code_block', '')

            vect = TfidfVectorizer().fit([ac, ans, ref])
            vecs = vect.transform([ac, ans, ref])
            mat = cosine_similarity(vecs)

            sim_ac_ans = mat[0, 1]
            sim_ac_ref = mat[0, 2]
            sim_ans_ref = mat[1, 2]

            writer.writerow([sid, f"{sim_ac_ans:.4f}", f"{sim_ac_ref:.4f}", f"{sim_ans_ref:.4f}"])
            sims_ac_ans.append(sim_ac_ans)
            sims_ac_ref.append(sim_ac_ref)
            sims_ans_ref.append(sim_ans_ref)

        # 平均值
        writer.writerow([
            'average',
            f"{(sum(sims_ac_ans)/len(sims_ac_ans)):.4f}",
            f"{(sum(sims_ac_ref)/len(sims_ac_ref)):.4f}",
            f"{(sum(sims_ans_ref)/len(sims_ans_ref)):.4f}"
        ])


def compute_jaccard_similarities(data, csv_path):
    """
    计算基于词元集的 Jaccard 相似度
    :param data: JSON 数据列表
    :param csv_path: 输出 CSV 文件路径
    """
    def jaccard(str1, str2):
        tokens1 = set(re.findall(r"\w+", str1))
        tokens2 = set(re.findall(r"\w+", str2))
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['submission_id', 'jacc_AC_ANS', 'jacc_AC_REF', 'jacc_ANS_REF'])
        sims_ac_ans, sims_ac_ref, sims_ans_ref = [], [], []

        for item in data:
            sid = item.get('submission_id')
            ac, ans, ref = item.get('sourceCode', ''), item.get('generate_code_block', ''), item.get('generate_ref_code_block', '')

            j_acc_ans = jaccard(ac, ans)
            j_acc_ref = jaccard(ac, ref)
            j_ans_ref = jaccard(ans, ref)

            writer.writerow([sid, f"{j_acc_ans:.4f}", f"{j_acc_ref:.4f}", f"{j_ans_ref:.4f}"])
            sims_ac_ans.append(j_acc_ans)
            sims_ac_ref.append(j_acc_ref)
            sims_ans_ref.append(j_ans_ref)

        # 平均值
        writer.writerow([
            'average',
            f"{(sum(sims_ac_ans)/len(sims_ac_ans)):.4f}",
            f"{(sum(sims_ac_ref)/len(sims_ac_ref)):.4f}",
            f"{(sum(sims_ans_ref)/len(sims_ans_ref)):.4f}"
        ])


if __name__ == '__main__':
    input_file="deepseek_32b_cpp_extract.json"
    #output_file_c="qwen_32b_python_sim_cosine.csv"
    output_file_j = "deepseek_32b_cpp_sim_jaccard.csv"
    # 读取 JSON 文件一次
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    #compute_cosine_similarities(data, output_file_c)

    compute_jaccard_similarities(data, output_file_j)


