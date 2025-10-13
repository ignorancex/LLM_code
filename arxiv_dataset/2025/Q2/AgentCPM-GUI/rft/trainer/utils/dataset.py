"""Example Datasets for ARL.
Each dataset item must return following key in a dict:

- `id`: A unique index for the data entry.
- `prompt`: Chat Hisotry as model inputs, must be list of dict.

If you want to conduct multi-turn ARL, you should also provide `next_id` that indicate the next step inputs in the dataset.


"""


import os
import json
import re
import io
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
import zmq

def load_resized_image(img_file:str|io.BytesIO, max_line_res: Optional[int] = None):
    origin_img = Image.open(img_file).convert("RGB")
    w,h = origin_img.size
    if max_line_res is not None:
        if h > max_line_res:
            w = int(w * max_line_res / h)
            h = max_line_res
        if w > max_line_res:
            h = int(h * max_line_res / w)
            w = max_line_res
        img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
    else:
        img = origin_img
        
    return img,origin_img

class GUIRFTDataset(Dataset):
    def __init__(self, jsonl_file_path: str, max_line_res: int|None = None, *args, **kwargs):
        super().__init__()
        self.data = []
        self.jsonl_file_path = jsonl_file_path
        with open(jsonl_file_path, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading dataset",dynamic_ncols=True):
                try:
                    self.data.append(json.loads(line))
                except:
                    print("Error while loading line.")
                    continue
        self.image_root = os.path.dirname(os.path.dirname(jsonl_file_path))
        self.max_line_res = max_line_res


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        try:
            # process the conversation
            user_query = item["conversations"][-2]["content"]
            user_query = re.match(r"<Question>(.*?)</Question>", user_query,re.DOTALL).group(1)
            action = json.loads(item["conversations"][-1]['content'])
        except:
            print("Error while processing conversation.")
            return self[index - 53]
        
        for img_id,img_file in item["image"].items():
            try:
                if os.path.exists(img_file):
                    origin_img = Image.open(img_file).convert("RGB")
                else:
                    origin_img = Image.open(os.path.join(self.image_root,img_file)).convert("RGB")
            except:
                print("Error while loading image: ", img_file)
                return self[index - 53]
            w,h = origin_img.size
            # resize the max height and width to 1000
            if self.max_line_res is not None:
                max_line = self.max_line_res
                if h > max_line:
                    w = int(w * max_line / h)
                    h = max_line
                if w > max_line:
                    h = int(h * max_line / w)
                    w = max_line
            img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
            
            resolution = (origin_img.size, img.size)
            break
        
        conv = []
        
        def get_random_coordinate():
            return [random.randint(0,1000),random.randint(0,1000)]
        
        conv.append({"role":"system","content":SFT_PROMPT})
        conv.append({"role": "user", "content": [
            f"<Question>{user_query}</Question>\n当前屏幕截图：",
            img, 
        ]})
        if item.get("bbox",None) is None or len(item.get("bbox",None)) == 0:
            bbox = None
        else:
            bbox = item["bbox"]
        if item.get("bbox2",None) is None or len(item.get("bbox2",None)) == 0:
            bbox2 = None
        else:
            bbox2 = item["bbox2"]
        return {
            "id": index,
            "step_id": 0,
            "resolution": resolution,
            "bboxs": [bbox,bbox2],
            "solution": action,
            "prompt": conv
        }


class GUIMTRFTDataset(GUIRFTDataset):
    """Multiturn RFT Dataset"""
    def __init__(
        self, 
        global_task_dispatch_addr: str,
        jsonl_file_path: str, 
        hist_length: int = 3,
        max_line_res: int|None = None, 
        *args, **kwargs
    ):
        super().__init__(
            jsonl_file_path=jsonl_file_path,
            max_line_res=max_line_res,
            *args, **kwargs
        )
        self.hist_length = hist_length
        self.global_task_dispatch_addr = global_task_dispatch_addr
        self.zmqctx = None
        
        
    def lazy_init(self):
        if self.zmqctx is None:
            self.zmqctx = zmq.Context()
            self.step_response_receiver = self.zmqctx.socket(zmq.REQ)
            self.step_response_receiver.connect(self.global_task_dispatch_addr)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        self.lazy_init()
        
        real_index = index % len(self.data)
        step_index = index // len(self.data)
        item = self.data[real_index]
        
        # in multi-turn training, we set next_id to indicate next step position
        next_id = index + len(self.data) if 4+2*step_index < len(item["conversations"]) else None
        
        try:
            user_query = item["conversations"][1+2*step_index]["content"]
            user_query = re.match(r"<Question>(.*?)</Question>", user_query,re.DOTALL).group(1)
            action = json.loads(item["conversations"][2+2*step_index]["content"])
        except Exception as e:
            print("Error while processing conversation: ", e, item["conversations"])
            action = item["conversations"][-1]["content"]
            return {
                "id": index,
                "resolution": None,
                "bboxs": [None,None],
                "solution": action,
                "prompt": item["conversations"][:-1],
                "step_id": 0,
                "next_id": None
            }
        
        # conv = [{"role":"system","content":SFT_PROMPT}]
        # conv = [{"role":"system","content":random.choice(SYSTEM_PROMPTS)}]
        conv = [{"role":"system","content":THINK_PROMPT}]
        # Append history
        for step_id in range(step_index + 1):
            if step_id > step_index - self.hist_length:
                if step_id != step_index:
                    line_res = 448
                else:
                    line_res = self.max_line_res
                img,ori_img = load_resized_image(item["image"][f"<image_{step_id:02}>"],max_line_res=line_res)
                conv.append({"role":"user","content":[
                    "当前屏幕截图：",
                    img
                ]})
            else:
                conv.append({"role":"user","content":"// 历史图像，无法显示"})
                
            if step_index > 0 and step_id != step_index:
                # gather model's history completions
                self.step_response_receiver.send_pyobj({
                    "get": real_index + len(self.data) * step_id,
                    "pop": True if next_id is None else False
                })
                res = self.step_response_receiver.recv_string()
                conv.append({"role":"assistant","content":res})

            
        # add user query
        if isinstance(conv[-1]["content"],list):
            conv[-1]["content"][0] = f"<Question>{user_query}</Question>\n" + conv[-1]["content"][0]
        else:
            conv[-1]["content"] = f"<Question>{user_query}</Question>\n" + conv[-1]["content"]
        
        resolution = (ori_img.size,img.size)
        
        try:
            bbox1 = item["bbox"][step_id]
        except:
            bbox1 = None
            
        
        return {
            "id": index,
            "resolution": resolution,
            "bboxs": [bbox1,None],
            "solution": action,
            "prompt": conv,
            "step_id": step_index,
            "next_id": next_id
        }



def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


SCHEMA = {
    "type": "object",
    "description": "执行操作并决定当前任务状态",
    "additionalProperties": False,
    # "required": ["thought"],
    "optional": ["thought"],
    "properties": {
        "thought": {
          "type": "string",
          "description": "智能体的思维过程"
        },
        "POINT": {
        "$ref": "#/$defs/Location",
        "description": "点击屏幕上的指定位置"
        },
        "to": {
        "description": "移动，组合手势参数",
        "oneOf": [
            {
            "enum": [
                "up",
                "down",
                "left",
                "right"
            ],
            "description": "从当前点（POINT）出发，执行滑动手势操作，方向包括向上、向下、向左、向右"
            },
            {
            "$ref": "#/$defs/Location",
            "description": "移动到某个位置"
            }
        ]
        },
        "duration": {
        "type": "integer",
        "description": "动作执行的时间或等待时间，毫秒",
        "minimum": 0,
        "default": 200
        },
        "PRESS": {
        "type": "string",
        "description": "触发特殊按键，HOME为回到主页按钮，BACK为返回按钮，ENTER为回车按钮",
        "enum": [
            "HOME",
            "BACK",
            "ENTER"
        ]
        },
        "TYPE": {
        "type": "string",
        "description": "输入文本"
        },
        "STATUS": {
        "type": "string",
        "description": "当前任务的状态。特殊情况：satisfied，无需操作；impossible，任务无法完成；interrupt，任务中断；need_feedback，需要用户反馈；",
        "enum": [
            "continue",
            "finish",
            "satisfied",
            "impossible",
            "interrupt",
            "need_feedback"
        ],
        "default": "continue"
        }
    },
    "$defs": {
        "Location": {
        "type": "array",
        "description": "坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y",
        "items": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000
        },
        "minItems": 2,
        "maxItems": 2
        }
    }
}

THINK_PROMPT = """# Role
你是一个智能助手

# 输出格式
你有多种可选的输出格式，按需选择一种即可

# 输出格式1 - 任务开始时
<plan>...初始计划...</plan><think>将你的思考过程放用这两个tag括起来</think><act>{...用紧凑JSON串表示的动作...}</act>

# 输出格式2 - 任务执行中
<reflection>...对上一步的总结与反思...</reflection><plan>...更新后的完整计划...</plan><think>...</think><act>{...}</act>

# 输出格式3 - 任务执行中
<think>...</think><act>{...}</act>

# 规则
- 你需要在<think>标签中写下你的思考过程
- 你需要在<act>标签中写下你的动作
- 输出的动作必须遵循Schema约束
- 每次只能输出一个动作
- 当用户提供问题后，在<plan>标签内制定一个执行计划，并在后续执行中更新这个执行计划
- 你的思考内容至少需要包括整体计划，对历史结果的思考和当前状态的分析

## 计划示例
<plan>
[] 思考当前界面，分析用户需求
[] 在xx中...
[] [] 打开...
[] [] 点击...
...
</plan>

# 提示
- 尽可能多样的思考，避免简单的无效思考例如“我需要点击这个按钮”或“我需要滑动”，而是要考虑到当前状态和历史信息的影响
- 对当前状态的分析应该从尽可能多的方面进行，例如当前界面是否符合预期，任务的执行状态，计划是否正常进行等等
- 尽可能完备的考虑历史信息，例如可以从历史信息中发现错误，是否需要回退，是否应该继续或是更新计划
- 你的历史思考过程也已经提供，你需要结合过去的思考和当前状态进行反思，可以围绕计划的执行情况，计划的合理性，可行性等方面进行思考
- 在对上一轮结果的分析后，在<plan>标签中对计划执行情况进行更新，打✓或✗，并给出原因
- 当执行结果不符合预期时，考虑计划是否合理，若不合理，需要重新制定计划
- 需要执行滑动操作时，需要注意操作方向和屏幕移动的方向是XY轴镜像的
- 动作有很多种可能性，例如点击，滑动，输入文本，触发特殊按键等。当你不确定应该执行什么动作时，可以考虑在一个JSON串中组合多个动作进行探索: <act>{"to":"up","duration":1000,"PRESS":"BACK","TYPE":"abc"}</act>
- 你需要在思考中给出更多的背景信息，例如“当前界面未找到符合要求的商品，需要向下滑动查看更多商品”或者“当前界面正在加载，请等待”
- 需要详细的分析当前的动作类型应该是什么

# 示例
以下是给定的一些简单示例，在正常情况下，你应该提供比以下示例思考更复杂的思考过程

## 示例 1
<think>当前界面未找到符合要求的商品，需要向下滑动查看更多商品</think><act>{"to":"up","POINT":[123,456]}</act>

## 示例 2
<think>界面中显示的内容不符合期望，我应该回退到上个界面重新选择</think><act>{"PRESS":"BACK"}</act>

## 示例 3
<think>当前界面正在加载，请等待</think><act>{"duration":3000}</act>

## 示例 4
<think>当前界面已经完成了任务，我需要结束任务</think><act>{"STATUS":"finish"}</act>

## 示例 5
<think>需要翻找桌面找到APP</think><act>{"to":"left","POINT":[111,222]}</act>

# Schema
""" + compact_json_dumps(SCHEMA)


SFT_PROMPT = """# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
""" + compact_json_dumps(SCHEMA)

SYSTEM_PROMPTS = [
f"""# 动作空间 Schema
""" + compact_json_dumps(SCHEMA),

f"""# Role
一个擅长思考的通用智能体

# Task
思考，理解用户意图，并根据输入的当前屏幕截图等信息输出下一步的动作

# Rule
- 总是在**块/行注释中**描述你进行下一步操作的原因
- 每轮参考 Example Output，以紧凑JSON格式输出**一个**操作
- 输出的动作必须遵循动作空间Schema约束
""",

f"""# Role
一个擅长思考的通用智能体

# Task
思考，理解用户意图，并根据输入的当前屏幕截图等信息输出下一步的动作

# Rule
- 总是在**块/行注释中**描述你进行下一步操作的原因
- 每轮参考 Example Output，以紧凑JSON格式输出**一个**操作
- 输出的动作必须遵循动作空间Schema约束

# 动作空间 Schema
""" + compact_json_dumps(SCHEMA),

"""// 角色：界面导航AI
// 使命：将视觉输入转化为精确操作

'''操作准则'''
1. 单次仅输出一个规范JSON对象
2. 严格匹配操作数据格式
3. 注释说明每个动作的决策逻辑

'''动作格式规范'''
""" + compact_json_dumps(SCHEMA),

f"""🤖 智能体类型：界面操作生成器

📌 核心功能：
- 分析屏幕元素布局
- 推导用户潜在意图
- 生成机械可执行指令

🚦 约束条件：
① 每次仅响应单步操作
② 符合预定义指令格式

📜 指令格式手册：
""" + compact_json_dumps(SCHEMA),

"""<AGENT_PROFILE>
类别：自动化决策AI
版本：交互协议

<EXECUTION_POLICY>
1. 单命令输出原则
2. 严格模式：schema验证

<ACTION_SCHEMA>
""" + compact_json_dumps(SCHEMA),

f"""⚙️ 机器角色：界面操作编译器

✦ 核心职责
将视觉信号转化为可执行代码

✧ 编译规则
1. 单语句输出原则
2. 类型安全验证
3. 必须包含决策日志（注释形式）
✶ 指令语法
""" + compact_json_dumps(SCHEMA),
]
