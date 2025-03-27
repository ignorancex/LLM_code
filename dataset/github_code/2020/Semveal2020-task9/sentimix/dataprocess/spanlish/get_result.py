
def process_text():
    ret_labels = []
    ret_sid = []
    with open("./Data/spang_test_conll_unlabel.txt", 'r', encoding='utf8') as my_file:

        label = ''
        s_id = ''

        for line in my_file.readlines():
            line = line.replace('\n', '').split()
            print(line)
            if (len(line) == 3):
                label = line[2]
                s_id = line[1]
                ret_labels.append(label)
                ret_sid.append(s_id)
        print(ret_labels,ret_sid)
    return  ret_labels, ret_sid

if __name__ == '__main__':
    ret_labels,ret_sid = process_text()
    with open("some_result.txt", 'w', encoding='utf8') as my_file:
        my_file.write('Uid'+","+'label\n')
        for i, label in enumerate(ret_labels):
            my_file.write('%s,%s\n' % (ret_sid[i],label ))