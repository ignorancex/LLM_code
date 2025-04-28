import json

json_file_path = "/home/user/asr-fuzzing-testing/CrossASRplus/examples/output/augmenter_full/result/google/wav2vec2_deepspeech_wav2letter/num_iteration_5/text_batch_size_1200/all_test_cases.json"
output_file_path = "./CrossASRplus/tests/augmenter_mutated_test_cases.json"

count = 0
all_mutated_test_cases = {}
# Opening JSON file
with open(json_file_path) as json_file:
    data = json.load(json_file)
 
    for key, value in data.items():
        all_mutated_test_cases[key] = []
        for test_case in data[key]:
            if test_case['text']['is_mutated']:
                all_mutated_test_cases[key].append(test_case)
                count += 1

    with open(output_file_path, 'w') as outfile:
            json.dump(all_mutated_test_cases, outfile, indent=2, sort_keys=True)

print(count)
