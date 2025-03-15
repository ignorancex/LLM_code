import numpy as np
import fasttext
model = fasttext.load_model("./saved_fasttext_model.bin")


score_list = {}
vector_output = np.array(model.get_output_matrix())
word_list = model.get_words()
input_matrix = model.get_input_matrix()
if(model.get_labels()[1] == "__label__1"):
    for i in range(0,len(model.get_words())):
        word = word_list[i]
        vector_input = input_matrix[i]
        result = np.matmul(np.array(vector_input).transpose(), vector_output.transpose())
        contribution = result[1] - result[0]
        score_list[word] = contribution

else:
    for i in range(0,len(model.get_words())):
        word = word_list[i]
        vector_input = input_matrix[i]
        result = np.matmul(np.array(vector_input).transpose(), vector_output.transpose())
        contribution = result[0] - result[1]
        score_list[word] = contribution

sorted_list = sorted(score_list.items(), key=lambda item: item[1],reverse=True)
reverse_sorted_list = sorted(score_list.items(), key=lambda item: item[1],reverse=False)
print(sorted_list[:50])
print("\n")
print(reverse_sorted_list[:50])