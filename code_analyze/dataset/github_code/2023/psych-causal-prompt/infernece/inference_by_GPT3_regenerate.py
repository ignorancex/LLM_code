import numpy as np
from icecream import ic
from tqdm import trange
import torch
import pandas as pd
from warping_dataset_like_GPT2 import GPT3TokenizerWarper, GPT3Warper, generate_prompts
import argparse
from datasets import load_from_disk
import numpy as np
from icecream import ic
from tqdm import trange
import torch
import pandas as pd
from warping_dataset_like_GPT2 import GPT3TokenizerWarper, GPT3Warper, generate_prompts
import argparse
from datasets import load_from_disk

if __name__=='__main__':
    setup = 2
    df = pd.read_csv('finalized_short_prompt_new.csv')
    perfix_list, postfix_list = generate_prompts(df, setup)

    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(GPT3TokenizerWarper([perfix_list[i], postfix_list[i]]))

    model = GPT3Warper("path2OpenAIKey", f"gpt3_result_setup_{setup}_short_large_regenerate2.jsonl")

    dataset = load_from_disk('./yelp_large_test_new')
    #ic(dataset[:]['text'])
    str_list = []
    idx = [4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4563, 4564, 4565, 8573, 8574, 8575, 8576, 8577, 8578, 8579, 8580, 8581, 8582, 8583, 8584, 8585, 8586, 8587, 8588, 8589, 8590, 8591, 8592, 8593, 8594, 8595, 8596, 8597, 8598, 8599, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8611, 8612, 8613, 8614, 8615, 8616, 8617, 8618, 8619, 8620, 8621, 8622, 8623, 8624, 8625, 8626, 8627, 8628, 8629, 8630, 8631, 8632, 8633, 8634, 8635, 8636, 8637, 8638, 8639, 8640, 8641, 8642, 8643, 8644, 8645, 8646, 8647, 8648, 8649, 8650, 8651, 8652, 8653, 8654, 8655, 8656, 8657, 8658, 8659, 8660, 8661, 8662, 8663, 8664, 8665, 8666, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674, 8675, 8676, 8677, 8678, 8679, 8680, 8681, 8682, 8683, 8684, 8685, 8686, 8687, 8688, 8689, 8690, 8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712]
    for i in idx:
        str_list.append(dataset[i+1000]['text'])
    wraped_list = []
    for i in range(len(perfix_list)):
        wraped_list.append(tokenizer_list[i](str_list))
    ic(len(perfix_list))
    ic(len(str_list))
    ic(len(wraped_list[0]))
    print("###############################")
    #print(wraped_list[0]['input_prompts'][0])
    ic(len(wraped_list[0]))
    result_arr = []
    for i in wraped_list:
        result_arr.append(model(**i))
    a = np.array(result_arr)
    #ic(a[0])
    np.save(f"gpt3_result_setup_{setup}_short_large_regenerate2.npy",a)
