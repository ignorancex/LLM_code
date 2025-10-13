import openai
import os, sys
from global_config import OPENAI_KEY


# Initialize the OpenAI API client
openai.api_key = OPENAI_KEY

length = "800~1300"

prompt = """\
根据表单标题创造出该纸本表单中可能出现的内容，其中包含多组键值对，请确保内容的真实性以及多样性，且表单使用地区为中国。键值对不限于一对一，也可使用一对多的勾选题，多样性越高越好。请不要使用「王小明」这种太过常见的名字。  
同一段落使用<h{num}></h{num}>包起来。  
每个实体使用"-"符号为开头列出，若生成的问题是一个小标题对应到更多键值对，请多用一个"-"符号标示，请严格遵守有层级关系的实体用"-"符号的数量区隔。  
请生成约 {length} 字的内容，生成内容请放置于<内容></内容>符号内，并用 plaintext 输出，不可使用 markdown、不可使用 tab。  
请假装你在填写这份纸本表单，所以需要一并填入答案，并确保答案看起来如同真实资料、多样性高、层级结构越复杂越好。"""

example_q = "华东师范大学新生入学健康检查表"

example_a = """\
<内容>
<h1> 个人基本资料
- 姓名: 陈淑芬
- 性别:
-- ☐ 男
-- ☑ 女
- 出生日期: 1999年5月30日
- 身份证号码: A123456789
- 联系电话: 0987654321
- 紧急联系人信息
-- 紧急联系人1:
--- 姓名: 林建勋
--- 关系: 父亲
--- 电话: 0978654321
-- 紧急联系人2:
--- 姓名: 陈建宏
--- 关系: 爷爷
--- 电话: 0912587465
</h1>

<h2> 常见家族病史
- 请勾选家族中是否有以下疾病:
-- ☐ 糖尿病
-- ☑ 高血压
-- ☐ 心脏病
-- ☐ 癌症
-- ☑ 中风
- 其他 (请列出): 无
</h2>

<h3> 过去病史及目前病情
- 过去曾患疾病 (请勾选):
-- ☑ 水痘
-- ☐ 肝炎
-- ☐ 结核病
- 目前是否有持续追踪的病情:
-- ☐ 无
-- ☑ 有 (请说明):
--- 过敏性鼻炎，定期就诊中。
</h3>

- （以下由医院人员填写）

<h4> 近视及听力检查结果
- 视力检查:
-- 右眼 (矫正后): 1.0
-- 左眼 (矫正后): 1.0
- 听力检查:
-- 右耳: 正常
-- 左耳: 正常
</h4>

<h5> 血液常规检查报告
- 红血球计数 (RBC): 4.5 x10^6/μL
- 白血球计数 (WBC): 6.0 x10^3/μL
- 血红素 (Hgb): 13.5 g/dL
- 血小板计数 (PLT): 250 x10^3/μL
</h5>

<h6> 尿液常规检查结果
- 尿液颜色:
-- ☑ 浅黄
-- ☐ 深黄
-- ☐ 其他 (请说明):
- 尿液分析结果:
-- 蛋白质: 阴性
-- 葡萄糖: 阴性
-- 细菌: 阴性
</h6>

- 备注
-- 1. 建议定期进行健康检查。
-- 2. 注意饮食均衡，避免高盐高脂饮食。
-- 3. 保持规律运动，强化心肺功能。
-- 4. 如有任何不适，请立即就医。

- 学生签名: 陈淑芬
- 提交日期: 2023年10月3日
</内容>
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
        model="gpt-4o-mini",
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