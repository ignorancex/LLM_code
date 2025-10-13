import pandas as pd
import re

def simplify_text(text):
    if not isinstance(text, str):
        return text  # 处理空值或非字符串情况

    # 替换特定短语
    text = re.sub(r"\ba neighborhood of houses\b", "several buildings", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhouses|homes\b", "buildings", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhouse|home\b", "building", text, flags=re.IGNORECASE)
    text = re.sub(r"\bseems to be\b", "is", text, flags=re.IGNORECASE)

    if re.search(r"\bno buildings\b", text, re.IGNORECASE):
        return "no buildings."

    text = re.sub(r"\b(One of|Some of|Both)[^.]*\.", "", text, flags=re.IGNORECASE)

    # 提取包含关键词的句子
    sentences = re.findall(r'[^.]*?(?:building|buildings|density|distribution)[^.]*?\.', text, flags=re.IGNORECASE)

    processed_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\b(with|as|indicating|suggesting|showcasing|while)\b[^.]*\.', '.', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r',[^,]*(including|suggests|such as)[^.]*\.', '.', sentence, flags=re.IGNORECASE)

        if not re.search(r'\b(green-roofed|presence|city|rural|barn|appears|different|residential|separates|facilities|a shingled roof|suggests that)\b', sentence, re.IGNORECASE):
            sentence = sentence.strip()

            processed_sentences.append(sentence)

    # 句子数过多时，仅保留首尾句
    if len(processed_sentences) > 3:
        processed_sentences = [processed_sentences[0], processed_sentences[-1]]

    text = ' '.join(processed_sentences)

    # 删除冗余短语
    redundant_phrases = [
        "In the image, there is ",
        "In the image, there are ",
        "relatively ",
        "overall ",
        " in the background",
        "of buildings in the image ",
        "Additionally, there are ",
        "The buildings vary in size and shape.",
        " in the middle of a grassy field",
        "in the field ",
        "suggests that they ",
        "in the area ",
        "in the image "
    ]
    for phrase in redundant_phrases:
        text = text.replace(phrase, "").strip()

    text = re.sub(r',[^.]*\.', '.', text).strip()
    text = re.sub(r'\s+\.', '.', text)

    if text and not text.endswith('.'):
        text += '.'

    return text if text.strip() else " "

# 文件路径
# input_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\train\\Train_Text_A.xlsx"
# output_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\train\\Train_Text_A_O.xlsx"
# input_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\train\\Train_Text_B.xlsx"
# output_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\train\\Train_Text_B_O.xlsx"
# input_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\test\\Test_Text_A.xlsx"
# output_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\test\\Test_Text_A_O.xlsx"
input_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\test\\Test_Text_B.xlsx"
output_file = "X:\\WCM\\CMTG\\Dataset\\GZ\\test\\Test_TextB_O.xlsx"
df = pd.read_excel(input_file)

if 'Description' in df.columns:
    df['Description'] = df['Description'].apply(simplify_text)
    df.to_excel(output_file, index=False)
    print(f"文本精炼完成，已保存至 {output_file}")
else:
    print("Error: 'Description' 列未找到")
