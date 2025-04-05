node2text = dict()
f = open("/tmp/a", "r")
for line in f.readlines():
    items = line.strip().split("\t")
    if items[0]=="node":
        node2text[items[1]] = items[2]
    if items[0]=="edge":
        print (node2text[items[1]] + "\t" + items[2] + "\t" + node2text[items[3]])
