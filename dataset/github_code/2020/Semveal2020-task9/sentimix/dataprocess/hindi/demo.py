#
# import shutil
# readPath='./data/get_train.tsv'
# writePath='./data/onetrain.txt'
# lines_seen=set()
# outfiile=open(writePath,'a+',encoding='utf-8')
# f=open(readPath,'r',encoding='utf-8')
# k =1
# for line in f:
#     line = line.split('\t')[0]
#     # print(line)
#
#     if line not in lines_seen:
#             k=k+1
#             clean_sentence = []
#             review_text = line.split()
#             for i in review_text:
#                 if i.startswith("htt"):
#                     continue
#                 else:
#                     clean_sentence.append(i)
#             line = ' '.join(str(i) for i in clean_sentence)
#             outfiile.write(line+'\n')
#             lines_seen.add(line)
# print(k)
# print(lines_seen)
with open('demo.txt', 'w', encoding='utf8') as my_file:
    my_file.write('Uid'+","+'Sentiment\n')