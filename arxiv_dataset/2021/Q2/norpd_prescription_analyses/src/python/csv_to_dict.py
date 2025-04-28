import csv
def csv_to_dict(infile, delimiter = ";", inverse_map = False):
    d = {}
    with open(infile) as f:
        line = 0
        csv_reader = csv.reader(f, delimiter = delimiter)
        k_idx, v_idx = (0 , 1) if not inverse_map else (1 ,0)
        for row in csv_reader:
            d[row[k_idx]]=row[v_idx]
            line+=1
    return d