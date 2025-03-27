from pyrouge import Rouge155
import numpy

r = Rouge155()
rouge1_list = []
rouge2_list = []
rougeSU_list = []
totDocN = 5622
for i in range(totDocN):
    r.system_dir = 'system_summaries/' + str(i+1)
    r.model_dir = 'model_summaries/' + str(i+1)
    r.system_filename_pattern = 'text.(\d+).txt'
    r.model_filename_pattern = 'text.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    # print(output)
    output_dict = r.output_to_dict(output)
    rouge1_list.append(output_dict['rouge_1_f_score'])
    rouge2_list.append(output_dict['rouge_2_f_score'])
    rougeSU_list.append(output_dict['rouge_su*_f_score'])

# sorted_idx = numpy.argsort(rouge1_list)[::-1]
sorted_idx = range(totDocN)  # no sort

topN = 100
topR1Sum = 0
topR2Sum = 0
topRSUSum = 0
lastN = 100
lastR1Sum = 0
lastR2Sum = 0
lastRSUSum = 0
counter = 1
for idx in sorted_idx:
    print(idx, rouge1_list[idx], rouge2_list[idx], rougeSU_list[idx])
    if counter <= topN:
        topR1Sum += rouge1_list[idx]
        topR2Sum += rouge2_list[idx]
        topRSUSum += rougeSU_list[idx]
    elif counter > totDocN - lastN:
        lastR1Sum += rouge1_list[idx]
        lastR2Sum += rouge2_list[idx]
        lastRSUSum += rougeSU_list[idx]
    counter += 1

print('R1 Mean:', numpy.mean(rouge1_list), 'R2 Mean:', numpy.mean(rouge2_list), 'RSU Mean:', numpy.mean(rougeSU_list))
print('Top100 R1 Mean:', topR1Sum/topN, 'R2 Mean:', topR2Sum/topN, 'RSU Mean:', topRSUSum/topN)
print('Last100 R1 Mean:', lastR1Sum/topN, 'R2 Mean:', lastR2Sum/topN, 'RSU Mean:', lastRSUSum/topN)
