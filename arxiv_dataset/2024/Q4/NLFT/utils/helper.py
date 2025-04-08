sys = '''
Suppose you are a math expert. The following describes a math problem. Please read it carefully and solve it STEP BY STEP!!!, and give the correct answer.

Please ensure that your output strictly follows the following format requirements: {Your analysis} \n#### {The answer number}

Your analysis should be very detailed. And make sure the string "####" only appears once following the answer number in the end.

For example:
<Output Example>
Natalia sold 48\/2 = <<48\/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
</Output Example>
<Output Example>
The number of truck stamps is 11 + 9 = <<11+9=20>>20.\nThe number of rose stamps is 20 − 13 = <<20-13=7>>7.\nBella bought 11 + 20 + 7 = <<11+20+7=38>>38 stamps in all.\n#### 38
</Output Example>
<Output Example>
Lisa earned $60 * 1\/2 = $<<60*1\/2=30>>30.\nTommy earned $30 * 1\/2 = $<<30*1\/2=15>>15.\nLisa earned $30 - $15 = $<<30-15=15>>15 more than Tommy.\n#### 15
</Output Example>
<Output Example>
He needs to save up $400 because 4 x 100 = <<4*100=400>>400\nHe has 8 months to earn this money because 12 - 4 = <<12-4=8>>8\nHe needs to earn $50 a month because 400 \/ 8 = <<400\/8=50>>50\nHe needs to do 5 tasks a month because 50 \/ 10 = <<50\/10=5>>5\n#### 5
</Output Example>
<Output Example>
15 coins collected in hour one\n35 coins collected in hour two\n35 coins collected in hour three\n50 coins collected in hour four\nBefore giving her coworker some coins there were 15+35+35+50=<<15+35+35+50=135>>135 coins\nThe number of coins after given 15 to her coworker is 135-15=<<135-15=120>>120\n#### 120
</Output Example>

'''

sys2 = '''
Suppose you are a math expert and you are presented with a math problem, a student's response, and the correct answer. 

Please first check whether the student's response is correct. Including checking the solution process and whether the answer is correct.

If the student's response is correct, please directly output: ### The response is correct. ###

If the response is wrong, please analyze why the solution is wrong.
'''

def clear_gpu_memory():
    import torch
    torch.cuda.empty_cache()
    print("GPU内存已清理")

def clear_memory():
    import gc
    gc.collect()
    print("内存已清理")

# https://github.com/Guangxuan-Xiao/GSM8K-eval/blob/main/main.py
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = '[invalid]'
ANSWER_TRIGGER = "####"

# 用于提取 gsm8k output 里的 answer，很干净，不用鲁棒
def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "") # 20,000 => 20000
        return match_str
    else:
        assert False # 没道理走 else 分支
        return INVALID_ANS

# def is_correct(model_answer, answer):
#     gt_answer = extract_answer_from_output(answer)
#     assert gt_answer != INVALID_ANS
#     return model_answer == gt_answer

# 用于比较脏的提取 answer
def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER)
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

