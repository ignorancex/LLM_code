with open('spang_test_conll_unlabeled.txt', 'r', encoding='utf8') as my_file:
    text = ''
    label = ''
    s_id = ''
    i=0
    http_flag = False
    for line in my_file.readlines():
        line = line.replace('\n', '').split()

        if (len(line) == 3):
            i=i+1
            print(i)
    print(i)