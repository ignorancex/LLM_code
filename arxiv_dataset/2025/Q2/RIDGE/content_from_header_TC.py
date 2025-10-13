import openai
import os, sys
from global_config import OPENAI_KEY


# Initialize the OpenAI API client
openai.api_key = OPENAI_KEY

length = "800~1300"

prompt = """\
根據表單標題創造出該紙本表單中可能出現的內容，其中包含多組鍵值對，請確保內容的真實性以及多樣性，且表單使用地區為台灣。鍵值對不限於一對一，也可使用一對多的勾選題，多樣性越高越好。請不要使用「王小明」這種太過常見的名字。
同一段落使用<h{num}></h{num}>包起來。
每個實體使用"-"符號為開頭列出，若生成的問題是一個小標題對應到更多鍵值對，請多用一個"-"符號標示，請嚴格遵守有層級關係的實體用"-"符號的數量區隔。
請生成約 {length} 字的內容，生成內容請放置於<內容></內容>符號內，並用 plaintext 輸出，不可使用 markdown、不可使用 tab。
請假裝你在填寫這份紙本表單，所以需要一併填入答案，並確保答案看起來如同真實資料、多樣性高、層級結構越複雜越好。"""

example_q = "臺灣大學新生入學健康檢查表"

example_a = """\
<內容>
<h1> 個人基本資料
- 姓名: 陳淑芬
- 性別:
-- ☐ 男
-- ☑ 女
- 出生日期: 1999年5月30日
- 身分證字號: A123456789
- 聯絡電話: 0987654321
- 緊急聯絡人資訊
-- 緊急聯絡人1:
--- 姓名: 林建勳
--- 關係: 父親
--- 電話: 0978654321
-- 緊急聯絡人2:
--- 姓名: 陳建宏
--- 關係: 爺爺
--- 電話: 0912587465
</h1>

<h2> 常見家族病史
- 請勾選家族中是否有以下疾病:
-- ☐ 糖尿病
-- ☑ 高血壓
-- ☐ 心臟病
-- ☐ 癌症
-- ☑ 中風
- 其他 (請列出): 無
</h2>

<h3> 過去病史及目前病情
- 過去曾患疾病 (請勾選):
-- ☑ 水痘
-- ☐ 肝炎
-- ☐ 結核病
- 目前是否有持續追蹤的病情:
-- ☐ 無
-- ☑ 有 (請說明):
--- 過敏性鼻炎，定期就診中。
</h3>

- （以下由醫院人員填寫）

<h4> 近視及聽力檢查結果
- 視力檢查:
-- 右眼 (矯正後): 1.0
-- 左眼 (矯正後): 1.0
- 聽力檢查:
-- 右耳: 正常
-- 左耳: 正常
</h4>

<h5> 血液常規檢查報告
- 紅血球計數 (RBC): 4.5 x10^6/μL
- 白血球計數 (WBC): 6.0 x10^3/μL
- 血紅素 (Hgb): 13.5 g/dL
- 血小板計數 (PLT): 250 x10^3/μL
</h5>

<h6> 胸部X光檢查報告
- 檢查結果:
-- ☑ 正常
-- ☐ 異常 (請說明):
</h6>

<h7> 心電圖檢查報告
- 檢查結果:
-- ☑ 正常
-- ☐ 異常 (請說明):
</h7>

<h8> 肝功能檢查報告
- 檢查項目及結果:
-- 門冬氨酸轉氨酵素 (AST): 20 U/L
-- 丙氨酸轉氨酵素 (ALT): 25 U/L
</h8>

<h9> 尿液常規檢查結果
- 尿液顏色:
-- ☑ 淺黃
-- ☐ 深黃
-- ☐ 其他 (請說明):
- 尿液分析結果:
-- 蛋白質: 陰性
-- 葡萄糖: 陰性
-- 細菌: 陰性
</h9>

<h10> 疫苗接種記錄
- 是否已接種以下疫苗:
-- B型肝炎疫苗:
--- ☑ 是 (完成全程)
--- ☐ 否
-- 流感疫苗:
--- ☑ 是
--- ☐ 否
-- 破傷風疫苗:
--- ☑ 是
--- ☐ 否
- 其他疫苗:
-- MMR疫苗(麻疹、腮腺炎、德國麻疹):
--- ☑ 是
--- ☐ 否
</h10>

<h11> 醫生建議與評定
- 醫生建議:
-- 多進行規律運動，保持均衡飲食。
- 參檢醫師: 李光泉醫師
- 簽名: 李光泉
- 評定日期: 2023年10月1日
</h11>

- 備註
-- 1. 建議定期進行健康檢查。
-- 2. 注意飲食均衡，避免高鹽高脂飲食。
-- 3. 保持規律運動，強化心肺功能。
-- 4. 如有任何不適，請立即就醫。

- 學生簽名: 陳淑芬
- 繳交日期: 2023年10月3日
</內容>
"""

input_file_path = f"headers/{sys.argv[1]}.txt"
output_dir = f"contents/{os.path.basename(input_file_path).split('.')[0]}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_file =  open(input_file_path, "r", encoding="utf8")
headers = input_file.read().splitlines()
num_file = len(headers)
input_file.close()

for file_id in range(0, num_file):
    header = headers[file_id]
    header = header.split(".")[-1].strip()
    print(header)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(length=length, num="{num}")},
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
            {"role": "user", "content": header}
        ],
    )
    response = response.choices[0].message.content
    print(response)

    with open(f'{output_dir}/{file_id:06d}.txt', 'w', encoding="utf8") as f:
        f.write(header + '\n' + response)