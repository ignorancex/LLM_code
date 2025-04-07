#coding=utf8

INPUT_FILE = "../Multi-News/data/inputs.txt"
OUTPUT_FILE = "./input.urls"
URL_MAP = "./input.url_map"

if __name__ == '__main__':
    import sys, json
    fpout = open(OUTPUT_FILE, 'w')
    mpout = open(URL_MAP, 'w')
    url_map = {}
    for line in open(INPUT_FILE):
        flist = line.strip().split('\t')
        url = flist[0]
        uid = flist[1]
        ulist = url.split('/')
        url_timestamp = ulist[4][:10]+'id_'
        url = '/'.join(ulist[:4] + [url_timestamp] + ulist[5:])
        url_map[url] = uid
        fpout.write(url + '\n')
    mpout.write(json.dumps(url_map))
    fpout.close()
    mpout.close()
