import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_code_similarities(json_path, csv_path):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备 CSV 写入器
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['submission_id', 'sim_AC_ANS', 'sim_AC_REF', 'sim_ANS_REF'])

        # 存储所有相似度以便计算平均值
        sims_ac_ans = []
        sims_ac_ref = []
        sims_ans_ref = []

        # 对每个条目计算相似度
        for item in data:
            sid = item.get('submission_id')
            ac_code = item.get('sourceCode', '')
            ans_code = item.get('generate_code_block', '')
            ref_code = item.get('generate_ref_code_block', '')

            # 使用 TfidfVectorizer 将代码转换为向量
            vect = TfidfVectorizer().fit([ac_code, ans_code, ref_code])
            vecs = vect.transform([ac_code, ans_code, ref_code])

            # 计算余弦相似度矩阵
            sim_matrix = cosine_similarity(vecs)
            sim_ac_ans = sim_matrix[0, 1]
            sim_ac_ref = sim_matrix[0, 2]
            sim_ans_ref = sim_matrix[1, 2]

            # 写入当前行
            writer.writerow([sid, f"{sim_ac_ans:.4f}", f"{sim_ac_ref:.4f}", f"{sim_ans_ref:.4f}"])

            # 收集数值
            sims_ac_ans.append(sim_ac_ans)
            sims_ac_ref.append(sim_ac_ref)
            sims_ans_ref.append(sim_ans_ref)

        # 计算平均值
        avg_ac_ans = sum(sims_ac_ans) / len(sims_ac_ans) if sims_ac_ans else 0.0
        avg_ac_ref = sum(sims_ac_ref) / len(sims_ac_ref) if sims_ac_ref else 0.0
        avg_ans_ref = sum(sims_ans_ref) / len(sims_ans_ref) if sims_ans_ref else 0.0

        # 写入平均值行
        writer.writerow(['average', f"{avg_ac_ans:.4f}", f"{avg_ac_ref:.4f}", f"{avg_ans_ref:.4f}"])

if __name__ == '__main__':
    # 示例调用
    compute_code_similarities('deepseek_32b_cpp_extract.json', 'deepseek_32b_cpp_sim_cosine.csv')
    print("计算完成，结果已保存到 similarities.csv")
