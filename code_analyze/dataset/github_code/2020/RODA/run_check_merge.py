# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time, json
import torch.optim
from src.expressions_transfer import *

batch_size = 128
embedding_size = 128
hidden_size = 512
n_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

data = load_raw_data("data/Math_23K1.json")

with open("data/checkmerge.json", "rb") as f:
    merge_data = json.load(f)

with open("data/PreprocessedQuestion_enumeratefilteredtest2.json", "rb") as f:
    test_data = json.load(f)

with open("data/PreprocessedQuestion_enumeratefilteredvalid2.json", "rb") as f:
    dev_data = json.load(f)

look_up_table = [0 for i in range(len(merge_data) + 23162)]
test_table = [0 for i in range(len(test_data))]
dev_table = [0 for i in range(len(dev_data))]

for i, item in enumerate(merge_data):
    if "origin_id" in item.keys():
        look_up_table[i+23162] = int(item["origin_id"]) - 1
#    else:
#        print('-', end='')
for i, item in enumerate(test_data):
    if "origin_id" in item.keys():
        test_table[i] = int(item["origin_id"]) - 1

for i, item in enumerate(dev_data):
    if "origin_id" in item.keys():
        dev_table[i] = int(item["origin_id"]) - 1

del merge_data, test_data, dev_data

look_up_table = look_up_table + test_table + dev_table

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

fold_size = int(23162 * 0.2)
fold_pairs = []
test_pairs = []
augment_data = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
    test_pairs.append(pairs[fold_start:fold_end])
    augment_data.append([pairs[i] for i in range(23162, len(pairs)) if (not fold_start< look_up_table[i] < fold_end and look_up_table[i] != 0)])
fold_pairs.append(pairs[(fold_size * 4): 23162])
test_pairs.append(pairs[(fold_size * 4): 23162])
print(fold_pairs == test_pairs)
augment_data.append([pairs[i] for i in range(23162, len(pairs)) if (not fold_start< look_up_table[i] <23162 and look_up_table[i] != 0)])

print(len(fold_pairs), len(fold_pairs[0]),len(fold_pairs[1]),len(fold_pairs[2]),len(fold_pairs[3]),len(fold_pairs[4]))

print(len(augment_data), len(augment_data[0]),len(augment_data[1]),len(augment_data[2]),len(augment_data[3]),len(augment_data[4]))

best_acc_fold = []

for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    pairs_trained += augment_data[fold]#[0: int(len(augment_data[fold]) * 0.5)]
    print(len(pairs_trained), len(pairs_tested))
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,  copy_nums, tree=True)
    print( len(train_pairs), len(test_pairs))
# Initialize models

    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    for epoch in range(n_epochs):
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        for idx in range(len(input_lengths)):
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx])
            loss_total += loss

        print("loss:", loss_total / len(input_lengths))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        if epoch % 10 == 0 or epoch > n_epochs - 5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pairs:
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], beam_size=beam_size)
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            torch.save(encoder.state_dict(), "models/encoder")
            torch.save(predict.state_dict(), "models/predict")
            torch.save(generate.state_dict(), "models/generate")
            torch.save(merge.state_dict(), "models/merge")
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))


